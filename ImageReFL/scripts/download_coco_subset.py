import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen

import datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download COCO subset images by index")
    parser.add_argument("--dataset", default="phiyodr/coco2017")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--subset-size", type=int, default=1000)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-column", default="captions")
    parser.add_argument("--image-column", default="coco_url")
    parser.add_argument("--save-captions", action="store_true")
    parser.add_argument("--captions-out", default=None)
    return parser.parse_args()


def get_image_name_by_index(index: int) -> str:
    return f"{index:05}.jpg"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = f"{args.split}[:{args.subset_size}]"
    ds = datasets.load_dataset(
        args.dataset,
        split=split,
        trust_remote_code=True,
        streaming=False,
    )

    captions = []
    for idx in tqdm(range(len(ds)), desc="download"):
        row = ds[idx]
        url = row.get(args.image_column)
        if not url:
            raise RuntimeError(f"Missing {args.image_column} at index {idx}")
        img_path = out_dir / get_image_name_by_index(idx)
        if not img_path.exists():
            with urlopen(url) as resp, open(img_path, "wb") as f:
                f.write(resp.read())
        if args.save_captions:
            caption = row.get(args.text_column)
            if isinstance(caption, list):
                caption = caption[0] if caption else ""
            captions.append({"prompt": caption})

    if args.save_captions:
        captions_out = args.captions_out or (out_dir.parent / "prompts.json")
        with open(captions_out, "w", encoding="utf-8") as f:
            json.dump(captions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
