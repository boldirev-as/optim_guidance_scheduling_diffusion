import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ParquetPromptDataset(Dataset):
    """Reads a Parquet file with a `prompt` column and performs on‑the‑fly tokenisation."""

    def __init__(
            self,
            parquet_path: str,
            caption_col: str = "prompt",
            tokenizer=None,
            rm_tokenizer=None,
            max_len: int = 77,
            rm_max_len: int = 35,
            train_mode: bool = True,
            seed: int = 42,
    ):
        super().__init__()
        if train_mode:
            self.df = pd.read_parquet(parquet_path)
            self.caption_col = caption_col
        else:
            with open(parquet_path, "r", encoding="utf-8") as f:
                val_list = json.load(f)
            self.val_prompts = [d["prompt"] for d in val_list]

        self.tokenizer = tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.max_len = max_len
        self.rm_max_len = rm_max_len
        self.train_mode = train_mode
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.df) if self.train_mode else len(self.val_prompts)

    def _pick_caption(self, raw_caption):
        if isinstance(raw_caption, str):
            return raw_caption
        if isinstance(raw_caption, (list, tuple, np.ndarray)):
            return random.choice(raw_caption) if self.train_mode else raw_caption[0]
        raise ValueError("Captions must be strings or lists of strings.")

    def __getitem__(self, idx: int):
        if self.train_mode:
            caption = self._pick_caption(self.df.iloc[idx][self.caption_col])
        else:
            caption = self.val_prompts[idx]

        txt_enc = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        rm_enc = self.rm_tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.rm_max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": txt_enc.input_ids.squeeze(0),  # (77,)
            "rm_input_ids": rm_enc.input_ids.squeeze(0),  # (35,)
            "rm_attention_mask": rm_enc.attention_mask.squeeze(0),
            "caption": caption,
        }
