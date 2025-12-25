import logging
import random
import typing as tp
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import IterableDataset
from huggingface_hub import hf_hub_download

from src.constants.dataset import DatasetColumns
from datasets import Image as HFImage
from src.utils.io_utils import get_image_name_by_index

logger = logging.getLogger(__name__)


def _load_text_only_dataset(
        *,
        # repo_id is optional; supports local files too
        repo_id: str | None,
        text_column: str,
        text_only_file: str,
        text_only_field: str | None = None,
        streaming: bool = True,
):
    """
    Грузим подписи/промпты из ОДНОГО локального файла или файла в HF Hub.
    Поддерживаются JSON/JSONL/Parquet/CSV. Для JSON можно передать:
      - root=[{...}] (list of dicts) с ключом 'prompt' (или др.)
      - {"field": [{...}]} через text_only_field.

    :param repo_id: HF Hub repo_id (если файл не локальный)
    :param text_column: имя колонки для нормализации текста
    :param text_only_file: путь к одному файлу (локально) или имя файла в репозитории HF Hub
    :param text_only_field: имя поля при структуре {"field": [...]}
    :param streaming: использовать ли стриминг (при отсутствии схемы будет авто-фоллбек)
    """
    filename = text_only_file

    # 1) Определяем источник: локальный путь или HF Hub
    candidate = Path(str(filename))
    if candidate.exists():
        local_path = str(candidate)
    else:
        if not repo_id:
            raise ValueError(
                f"File '{filename}' not found locally and repo_id is None; "
                f"provide a valid local path or set repo_id to fetch from HF Hub."
            )
        local_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # 2) Детектируем билдер
    if local_path.endswith((".jsonl", ".json")):
        builder = "json"
        kwargs = {"field": text_only_field} if text_only_field else {}
    elif local_path.endswith(".parquet"):
        builder = "parquet"
        kwargs = {}
    elif local_path.endswith(".csv"):
        builder = "csv"
        kwargs = {}
    else:
        raise ValueError(f"Unsupported annotation file: {local_path}")

    # Helper to actually load
    def _load(stream: bool):
        return datasets.load_dataset(
            builder,
            data_files=local_path,
            split="train",
            streaming=stream,
            **kwargs,
        )

    # 3) Загружаем датасет; если в streaming режиме нет column_names — перезагрузим без streaming
    ds = _load(streaming)
    colnames = getattr(ds, "column_names", None)

    if colnames is None:
        # Попробуем узнать схему через non-streaming
        logger.warning(
            "column_names is None in streaming mode for '%s'; reloading non-streaming to infer schema.",
            local_path,
        )
        ds_ns = _load(False)  # non-streaming
        colnames = getattr(ds_ns, "column_names", None)
        if colnames is None:
            # Последняя попытка — взять ключи первого примера
            try:
                first = ds_ns[0]
                colnames = list(first.keys())
            except Exception:
                pass
        # Используем non-streaming версию далее (надёжнее для нормализации колонок)
        ds = ds_ns

    if not colnames:
        raise RuntimeError(
            f"Failed to infer column names for '{local_path}'. "
            f"Consider using text_only_streaming=False or check file format."
        )

    # 4) Нормализуем текстовую колонку
    possible_keys = [text_column, "caption", "text", "prompt", "prompts"]
    have = set(colnames)
    key = next((k for k in possible_keys if k in have), None)
    if key is None:
        raise KeyError(f"Can't find text column among {have}; expected one of {possible_keys}")

    if key != text_column:
        # rename_column поддерживается и для обычного Dataset
        ds = ds.rename_column(key, text_column)

    # 5) Оставляем только текстовую колонку
    ds = ds.select_columns([text_column])
    return ds


class DatasetWrapper(Dataset):
    """
    Опциональная подгрузка изображений. Если load_images=False, пытаемся:
    1) stream через исходный билдер (если он поддерживает streaming без TAR),
    2) иначе fallback на text-only загрузку из *одного* файла аннотаций (локальный путь или HF Hub).

    Также поддерживается простой JSON-файл как список словарей с ключом 'prompt'.
    """

    def __init__(
            self,
            text_column: str,
            all_models_with_tokenizer: list,
            image_column: str | None = None,
            images_path: str | None = None,
            local_image_offset: int = 0,
            images_per_row: int | None = None,
            dataset_name: str | None = None,
            dataset_split: str | None = None,
            cache_dir: str | None = None,
            raw_dataset: datasets.Dataset | None = None,
            fixed_length: int | None = None,
            duplicate_count: int | None = None,
            resolution: int = 512,
            load_images: bool = True,
            use_one: bool = False,
            seed_range: int = 100,
            logger=None,

            # --- text-only / ОДИН ФАЙЛ ---
            text_only_repo_id: str | None = None,
            text_only_file: str | None = None,
            text_only_field: str | None = None,
            text_only_streaming: bool = False,

            # --- NEW: ограничение train на кусок COCO ---
            subset_percent: int | None = None,  # например 10 => [:10%]
            subset_size: int | None = None,  # например 50000 => [:50000]
            subset_seed: int = 0,  # зарезервировано под случайную выборку
    ):
        self.all_models_with_tokenizer = all_models_with_tokenizer
        self.text_column = text_column
        self.load_images = load_images

        # CHANGED: dataset_split теперь может быть типичной строкой HF (например "train" или "train[:10%]")
        self.dataset_split = dataset_split or "train"

        self.use_one = use_one
        self.seed_range = seed_range
        self.logger = logger or logging.getLogger(__name__)

        self.image_column = image_column if load_images else None
        self.images_path = Path(images_path) if (images_path and load_images) else None
        self.local_image_offset = local_image_offset if load_images else 0
        self.images_per_row = images_per_row if load_images else None
        self.fixed_length = fixed_length
        self.duplicate_count = duplicate_count
        self.resolution = resolution

        self.text_only_repo_id = text_only_repo_id
        self.text_only_file = text_only_file
        self.text_only_field = text_only_field
        self.text_only_streaming = text_only_streaming

        # NEW: параметры сабсета
        self.subset_percent = subset_percent
        self.subset_size = subset_size
        self.subset_seed = subset_seed

        # 1) Загрузка сырого датасета
        if raw_dataset is None:
            if not self.load_images:
                # text-only режим (как было)
                if dataset_name is None and self.text_only_file:
                    raw_dataset = _load_text_only_dataset(
                        repo_id=self.text_only_repo_id,
                        text_column=self.text_column,
                        text_only_file=self.text_only_file,
                        text_only_field=self.text_only_field,
                        streaming=self.text_only_streaming,
                    )
                else:
                    # пробуем стримить исходный билдер
                    try:
                        raw_dataset = datasets.load_dataset(
                            dataset_name,
                            split=self.dataset_split,
                            cache_dir=cache_dir,
                            trust_remote_code=True,
                            streaming=True,
                        )
                    except NotImplementedError as e:
                        msg = str(e)
                        tar_streaming = "iter_archive" in msg or "TAR archives" in msg
                        if tar_streaming:
                            logger.warning("Builder streaming not supported (TAR). Using text-only fallback.")
                            if not self.text_only_file:
                                raise RuntimeError(
                                    "Этот датасет не поддерживает streaming (TAR). "
                                    "Передай text_only_file (локальный путь или имя файла в HF) "
                                    "и опционально text_only_repo_id для загрузки только подписей."
                                ) from e
                            raw_dataset = _load_text_only_dataset(
                                repo_id=self.text_only_repo_id,
                                text_column=self.text_column,
                                text_only_file=self.text_only_file,
                                text_only_field=self.text_only_field,
                                streaming=self.text_only_streaming,
                            )
                        else:
                            raise
            else:
                # Полная загрузка с изображениями и RANDOM ACCESS.
                # NEW: аккуратно формируем split с учётом subset_*.
                effective_split = self._apply_subset_to_split(self.dataset_split)

                try:
                    raw_dataset = datasets.load_dataset(
                        dataset_name,
                        split=effective_split,
                        cache_dir=cache_dir,
                        trust_remote_code=True,
                        streaming=False,  # map-style для __getitem__
                    )
                except NotImplementedError as e:
                    # если билдера нет без streaming, пробуем streaming+materialize ближайший кусок
                    msg = str(e)
                    tar_streaming = "iter_archive" in msg or "TAR archives" in msg
                    if tar_streaming:
                        raise RuntimeError(
                            "Датасет хранится в TAR/ZIP архивах и требует скачивания целиком. "
                            "Выбери репозиторий на HF, где COCO лежит как imagefolder (по одному файлу) "
                            "или увеличь лимит диска."
                        ) from e
                    else:
                        raise

        if self.load_images and self.images_path is None:
            col = self.image_column
            names = getattr(raw_dataset, "column_names", [])
            if not col:
                col = "image" if "image" in names else None
            if col and isinstance(raw_dataset, datasets.Dataset):
                try:
                    # если колонка была строкой (путь/URL) — приведём к Image-фиче
                    raw_dataset = raw_dataset.cast_column(col, HFImage())
                except Exception:
                    # не страшно, ниже в _get_image обработаем вручную
                    pass

        # 2) Валидация колонок
        self._assert_dataset_is_valid(
            raw_dataset=raw_dataset,
            text_column=text_column,
            image_column=self.image_column if (self.load_images and (self.images_path is None)) else None,
        )

        self.raw_dataset = raw_dataset

        # 3) Трансформации изображений
        if self.load_images:
            self.image_process = transforms.Compose(
                [
                    transforms.Resize((self.resolution, self.resolution)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.image_process = None

    # NEW: конструируем slicing-строку для HF
    def _apply_subset_to_split(self, split: str) -> str:
        """
        Превращает 'train' в 'train[:10%]' или 'train[:50000]' если задан subset_*.
        Если split уже содержит слайс, оставляем как есть (считаем, что пользователь уверен).
        """
        if "[" in split and "]" in split:
            return split  # пользователь уже передал слайс явно

        if self.subset_percent is not None:
            assert 0 < self.subset_percent <= 100, "subset_percent должен быть в (0, 100]"
            return f"{split}[:{self.subset_percent}%]"

        if self.subset_size is not None:
            assert self.subset_size > 0, "subset_size должен быть > 0"
            return f"{split}[:{self.subset_size}]"

        return split

    def _get_caption(self, ind: int):
        if isinstance(self.raw_dataset, IterableDataset):
            sample = next(iter(self.raw_dataset.skip(ind).take(1)))
        else:
            sample = self.raw_dataset[ind]
        caption = sample[self.text_column]
        if isinstance(caption, list):
            caption = caption[0]
        return caption

    def _get_image(self, ind: int, image_index: int | None, original_index: int) -> torch.Tensor:
        # локальная папка с изображениями по индексу
        if self.images_path is not None:
            img = Image.open(
                self.images_path / get_image_name_by_index(original_index + self.local_image_offset)
            ).convert("RGB")
            return self.image_process(img).unsqueeze(0)

        # иначе — берём из датасета (колонка image должна быть PIL или list[PIL])
        data_dict = self.raw_dataset[ind]
        # если несколько картинок в примере
        if self.image_column and isinstance(data_dict[self.image_column], list):
            image_index = image_index or 0
            img = data_dict[self.image_column][image_index].convert("RGB")
        else:
            col = self.image_column or "image"
            img = data_dict[col].convert("RGB")
        return self.image_process(img).unsqueeze(0)

    def __getitem__(self, ind) -> dict[str, tp.Any]:
        original_index = ind
        base_ind = ind

        if self.use_one:
            base_ind = 10

        if self.duplicate_count is not None:
            base_ind //= self.duplicate_count

        image_index = None
        if self.load_images and self.image_column is not None and self.images_per_row is not None:
            image_index = base_ind % self.images_per_row
            base_ind //= self.images_per_row

        if self.use_one:
            caption = "girl in red coat on a rainy neon street, night, cinematic, highly detailed"
        else:
            caption = self._get_caption(base_ind)

        res = {"caption": caption}
        for model in self.all_models_with_tokenizer:
            res.update(model.tokenize(caption))

        if self.use_one:
            res["seeds"] = self.seed_range + ind

        if self.load_images:
            res[DatasetColumns.original_image.name] = self._get_image(
                ind=base_ind, image_index=image_index, original_index=original_index
            )
        return res

    def __len__(self):
        if self.use_one:
            return 100

        if self.fixed_length is not None:
            return self.fixed_length
        if isinstance(self.raw_dataset, IterableDataset):
            raise TypeError("Streaming dataset has unknown length; set fixed_length.")
        length = len(self.raw_dataset)
        if self.load_images and self.images_per_row is not None:
            length *= self.images_per_row
        if self.duplicate_count is not None:
            length *= self.duplicate_count
        return length  # CHANGED: теперь реальная длина

    @staticmethod
    def _assert_dataset_is_valid(
            raw_dataset: datasets.Dataset | IterableDataset,
            text_column: str,
            image_column: str | None = None,
    ) -> None:
        names = getattr(raw_dataset, "column_names", None)
        assert names and (text_column in names), f"text_column must be present in {names}"
        if not isinstance(raw_dataset, IterableDataset):
            first_row = raw_dataset[0]
            assert isinstance(first_row[text_column], str) or (
                    isinstance(first_row[text_column], list)
                    and first_row[text_column]
                    and isinstance(first_row[text_column][0], str)
            ), "text column must contain str or list[str]"
        if image_column is not None:
            assert (image_column in names), f"image_column must be present in {names}"
