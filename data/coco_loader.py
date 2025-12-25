import os
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CocoCaptions


def _prepare_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0,1]
    ])


def _get_caption(example) -> str:
    # Try several field names; pick first caption if list
    if "caption" in example:
        cap = example["caption"]
        if isinstance(cap, list):
            return cap[0]
        return cap
    if "captions" in example and example["captions"]:
        cap = example["captions"]
        if isinstance(cap, list):
            return cap[0]
        return cap
    # Fallback to empty string
    return ""


def _iter_hf(split: str):
    # Default HF COCO captions dataset; falls back to downloading if not present locally
    hf_split = "validation" if split.startswith("val") else "train"
    ds = load_dataset("HuggingFaceM4/COCO", "2017", split=hf_split)
    for ex in ds:
        img = ex.get("image")
        if isinstance(img, Image.Image):
            yield img, _get_caption(ex)


def _iter_torchvision(split: str, image_root: Path, ann_path: Path):
    dataset = CocoCaptions(
        root=str(image_root / ("val2017" if split.startswith("val") else "train2017")),
        annFile=str(ann_path),
    )
    for img, caps in dataset:
        cap = caps[0] if isinstance(caps, list) and caps else ""
        yield img, cap


def load_coco_data_batched(
    *,
    split: str = "val",
    num_samples: int = 1000,
    batch_size: int = 32,
    image_size: int = 512,
    image_root: str | None = None,
    ann_file: str | None = None,
) -> dict:
    """
    Load COCO images+captions in batches.

    Attempts torchvision CocoCaptions if image_root/ann_file provided or env vars
    COCO_IMAGE_ROOT / COCO_ANN_FILE are set; otherwise falls back to HuggingFace
    dataset HuggingFaceM4/COCO (downloads if missing).
    """
    transform = _prepare_transform(image_size)

    # Resolve paths from args/env
    image_root = image_root or os.getenv("COCO_IMAGE_ROOT")
    ann_file = ann_file or os.getenv("COCO_ANN_FILE")

    if image_root and ann_file and Path(image_root).exists() and Path(ann_file).exists():
        iterator = _iter_torchvision(split, Path(image_root), Path(ann_file))
    else:
        iterator = _iter_hf(split)

    batches: List[Tuple[torch.Tensor, List[str]]] = []
    imgs, caps = [], []
    count = 0
    for img, cap in iterator:
        imgs.append(transform(img))
        caps.append(cap)
        count += 1
        if len(imgs) == batch_size:
            batches.append((torch.stack(imgs), caps))
            imgs, caps = [], []
        if count >= num_samples:
            break

    # Flush remainder
    if imgs:
        batches.append((torch.stack(imgs), caps))

    return {
        "batches": batches,
        "num_samples": count,
        "batch_size": batch_size,
        "long_captions": [],
        "short_captions": [],
    }
