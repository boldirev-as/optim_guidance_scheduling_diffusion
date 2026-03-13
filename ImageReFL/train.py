import logging
import os
import warnings

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler

from src.models.stable_diffusion import GuidanceNet
from src.constants.trainer import TRAINER_NAME_TO_CLASS
from src.datasets.data_utils import get_dataloaders
from src.logger.noop import NoOpWriter
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="refl_train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    is_distributed = world_size > 1
    is_main_process = rank == 0

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed launch requires CUDA devices.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

    project_config = OmegaConf.to_container(config)
    if not is_distributed or is_main_process:
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(f"train_rank_{rank}")
        logger.setLevel(logging.INFO)
        writer = NoOpWriter()

    if is_distributed:
        device = f"cuda:{local_rank}"
    elif config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # build stable diffusion models
    model = instantiate(config.model).to(device)
    if model.use_ema:
        model.ema_unet.to(device)

    model.do_guidance_w_loss = config.model.do_guidance_w_loss

    # guidance_net = None
    # if config.model.do_guidance_w_loss:
    #     cond_dim = model.text_encoder.config.hidden_size
    #     guidance_net = GuidanceNet(cond_dim=cond_dim, n_t_steps=10, base_scale=7.5).to(device)

    # build reward models
    train_reward_model = instantiate(
        config.reward_models["train_model"], device=device
    ).to(device)
    train_reward_model.requires_grad_(False)

    val_reward_models = []
    for reward_model_config in config.reward_models["val_models"]:
        reward_model = instantiate(reward_model_config, device=device).to(device)
        reward_model.requires_grad_(False)
        val_reward_models.append(reward_model)

    val_model_metrics = []
    if config.reward_models.get("val_model_metrics", None) is not None:
        for reward_model_config in config.reward_models["val_model_metrics"]:
            reward_model = instantiate(reward_model_config, device=device).to(device)
            reward_model.requires_grad_(False)
            val_model_metrics.append(reward_model)

    all_models_with_tokenizer = val_reward_models + [model, train_reward_model]

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(
        config,
        device=device,
        all_models_with_tokenizer=all_models_with_tokenizer,
        logger=logger
    )

    if config.model.no_grad:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.guidance_net.parameters():
            p.requires_grad = True
        model.guidance_net.requires_grad_(True)

    # build optimizer, learning rate scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.guidance_net.parameters())
    optimizer = instantiate(config.optimizer, params=model.guidance_net.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    scaler = GradScaler()

    epoch_len = config.trainer.get("epoch_len")

    if config.trainer.type not in TRAINER_NAME_TO_CLASS:
        raise ValueError(f"Trainer type must be one of {TRAINER_NAME_TO_CLASS}")

    trainer_cls = TRAINER_NAME_TO_CLASS[config.trainer.type]

    trainer = trainer_cls(
        model=model,
        train_reward_model=train_reward_model,
        val_reward_models=val_reward_models,
        val_model_metrics=val_model_metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        # guidance_net=guidance_net
    )

    try:
        trainer.train()
    finally:
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
