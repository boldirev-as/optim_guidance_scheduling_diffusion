from itertools import repeat

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device: torch.device, all_models_with_tokenizer: list, logger):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = {
        dataset_partition: instantiate(
            config.datasets[dataset_partition],
            dataset_split=dataset_partition,
            all_models_with_tokenizer=all_models_with_tokenizer,
            logger=logger
        )
        for dataset_partition in config.datasets
    }

    leak_check_max = int(config.trainer.get("leak_check_max_samples", 0))
    leak_check_action = str(config.trainer.get("leak_check_action", "warn")).lower()
    if leak_check_max > 0 and "train" in datasets and "test" in datasets:
        train_ds = datasets["train"]
        test_ds = datasets["test"]

        def _collect_ids_or_captions(ds, max_samples):
            size = min(len(ds), max_samples)
            raw = getattr(ds, "raw_dataset", None)
            id_column = None
            if raw is not None:
                names = getattr(raw, "column_names", [])
                for candidate in ("image_id", "id"):
                    if candidate in names:
                        id_column = candidate
                        break

            items = set()
            for idx in range(size):
                try:
                    if id_column is not None:
                        sample = raw[idx]
                        items.add(int(sample[id_column]))
                        continue
                    caption = ds._get_caption(idx)
                except Exception:
                    break
                if isinstance(caption, str):
                    items.add(caption.strip().lower())
            return items, ("ids" if id_column is not None else "captions")

        train_items, train_mode = _collect_ids_or_captions(train_ds, leak_check_max)
        test_items, test_mode = _collect_ids_or_captions(test_ds, leak_check_max)
        overlap = train_items.intersection(test_items)
        if overlap:
            if train_mode == test_mode:
                mode = train_mode
            else:
                mode = "mixed"
            msg = (
                f"Leak check: found {len(overlap)} overlapping {mode} between train/test "
                f"in first {leak_check_max} samples."
            )
            if leak_check_action == "error":
                raise RuntimeError(msg)
            logger.warning(msg)
    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader[dataset_partition].batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader[dataset_partition],
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms
