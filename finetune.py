import argparse
import logging
import math
import os
import platform
import random
import sys

from dataclasses import dataclass
from typing import List, Dict, Any

from imscore.hps.model import HPSv2

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
import ImageReward as RM

BICUBIC = InterpolationMode.BICUBIC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("refl_trainer")

from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig


# 1) Structured schema (optional but catches typos & gives help)
@dataclass
class TrainConfig:
    # --- core ---
    grad_scale: float = 1e-3
    input_perturbation: float = 0.0
    revision: Optional[str] = None
    non_ema_revision: Optional[str] = None

    # --- data ---
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "text"
    max_train_samples: Optional[int] = None
    validation_prompts: List[str] = field(default_factory=list)

    # --- paths & I/O ---
    output_dir: str = "checkpoint/refl"
    cache_dir: Optional[str] = None
    logging_dir: str = "logs"

    # --- reproducibility ---
    seed: Optional[int] = None

    # --- image preprocessing ---
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False

    # --- training ---
    train_batch_size: int = 2
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = 100
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    base_lr: float = 5e-5
    learning_rate: float = base_lr * train_batch_size * gradient_accumulation_steps
    scale_lr: bool = False
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    snr_gamma: Optional[float] = None
    use_8bit_adam: bool = False
    allow_tf32: bool = False
    use_ema: bool = False

    # --- optimiser ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # --- misc/hub ---
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    report_to: str = "tensorboard"

    # --- distributed ---
    local_rank: int = -1

    # --- checkpointing ---
    checkpointing_steps: int = 100
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # --- memory/attention ---
    enable_xformers_memory_efficient_attention: bool = True
    noise_offset: float = 0.0

    # --- validation/logging ---
    validation_epochs: int = 5
    tracker_project_name: str = "text2image-refl"

    # --- dataloader ---
    dataloader_num_workers: int = 0

    mixed_precision: str = "fp16"


# 2) Helper replacing `parse_args`
def get_cfg() -> DictConfig:
    """Merge defaults, YAML file (if any) and CLI overrides."""
    # defaults from the dataclass
    cfg = OmegaConf.structured(TrainConfig)

    # optional YAML file:  python script.py --config=my_cfg.yaml
    cli = OmegaConf.from_cli()
    if "config" in cli:
        yaml_cfg = OmegaConf.load(cli.pop("config"))
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    # finally apply direct CLI overrides
    cfg = OmegaConf.merge(cfg, cli)

    # keep `non_ema_revision` in sync
    if cfg.non_ema_revision is None:
        cfg.non_ema_revision = cfg.revision

    # local-rank env override
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank

    return cfg


class ParquetPromptDataset(Dataset):
    """Reads a Parquet file with a `prompt` column and performs on‑the‑fly tokenisation."""

    def __init__(
            self,
            parquet_path: str,
            caption_col: str,
            tokenizer: CLIPTokenizer,
            rm_tokenizer: Any,
            max_len: int = 77,
            rm_max_len: int = 35,
            train_mode: bool = True,
            seed: int = 42,
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.caption_col = caption_col
        self.tokenizer = tokenizer
        self.rm_tokenizer = rm_tokenizer
        self.max_len = max_len
        self.rm_max_len = rm_max_len
        self.train_mode = train_mode
        random.seed(seed)

    def __len__(self) -> int:
        return len(self.df)

    def _pick_caption(self, raw_caption):
        if isinstance(raw_caption, str):
            return raw_caption
        if isinstance(raw_caption, (list, tuple, np.ndarray)):
            return random.choice(raw_caption) if self.train_mode else raw_caption[0]
        raise ValueError("Captions must be strings or lists of strings.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        caption = self._pick_caption(self.df.iloc[idx][self.caption_col])

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
            "caption": caption
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        if k == "caption":
            out[k] = [b[k] for b in batch]  # list of str
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


# ---------------------------------------------
#  Trainer
# ---------------------------------------------
class Trainer:
    def __init__(self, pretrained_model_name_or_path: str, train_parquet: str, args, device: str):
        self.device = device
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.args = args
        self.dtype = torch.float16

        if args.seed is not None:
            set_seed(args.seed)

        # sched / tokeniser / models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, device=self.device,
            dtype=self.dtype
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        ).half().to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
        ).half().to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        ).to(self.device)
        self.reward_model = RM.load("ImageReward-v1.0", device='cpu')  # stays fp32

        self.hps_model = HPSv2().to('cpu')

        self.hps_model.requires_grad_(False)

        # freeze non‑trainable
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.reward_model.requires_grad_(False)

        logger.info(
            "Trainable UNet params: %d",
            sum(p.numel() for p in self.unet.parameters() if p.requires_grad),
        )

        # EMA (optional)
        if args.use_ema:
            self.ema_unet = EMAModel(
                UNet2DConditionModel.from_pretrained(
                    pretrained_model_name_or_path, subfolder="unet", revision=args.revision
                ).parameters(),
                model_cls=UNet2DConditionModel,
                model_config=self.unet.config,
            )

        # memory‑efficient attention
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise RuntimeError("xformers not available, but requested.")

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            args.learning_rate *= args.gradient_accumulation_steps * args.train_batch_size

        # optimiser
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # dataset / loader
        self.train_dataset = ParquetPromptDataset(
            parquet_path=train_parquet,
            caption_col="prompt",
            tokenizer=self.tokenizer,
            rm_tokenizer=self.reward_model.blip.tokenizer,
            train_mode=True,
            seed=args.seed or 42,
        )
        if args.max_train_samples:
            self.train_dataset.df = self.train_dataset.df.sample(args.max_train_samples, random_state=args.seed)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

        # scheduler
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        args.max_train_steps = (
            args.num_train_epochs * self.num_update_steps_per_epoch
            if args.max_train_steps is None
            else args.max_train_steps
        )
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # dtype casting for inference‑only modules
        self.weight_dtype = torch.float32
        self.text_encoder.to(device, dtype=self.weight_dtype)
        self.vae.to(device, dtype=self.weight_dtype)
        # reward_model left in fp32

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train(self):
        args = self.args
        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info("Num examples           = %d", len(self.train_dataset))
        logger.info("Num epochs             = %d", args.num_train_epochs)
        logger.info("Instant batch size     = %d", args.train_batch_size)
        logger.info("Total batch size       = %d", total_batch_size)
        logger.info("Gradient accum steps   = %d", args.gradient_accumulation_steps)
        logger.info("Total optimisation steps = %d", args.max_train_steps)

        # progress_bar = tqdm(range(args.max_train_steps), desc="steps")
        global_step = 0

        rm_preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        for epoch in range(args.num_train_epochs):
            self.unet.train()
            for step, batch in enumerate(self.train_dataloader):
                batch_size = batch["input_ids"].shape[0]

                # text embeddings
                with torch.cuda.amp.autocast(dtype=self.weight_dtype):
                    encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.device))[0]

                # random latents
                latents = torch.randn((batch_size, 4, 64, 64), device=self.device, dtype=self.weight_dtype)

                # diffusion steps
                self.noise_scheduler.set_timesteps(40, device=self.device)
                mid_timestep = random.randint(30, 39)

                for t in self.noise_scheduler.timesteps[:mid_timestep]:
                    with torch.no_grad():
                        latent_in = self.noise_scheduler.scale_model_input(latents, t)
                        noise_pred = self.unet(latent_in, t, encoder_hidden_states=encoder_hidden_states).sample
                        latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

                t_mid = self.noise_scheduler.timesteps[mid_timestep]
                latent_in = self.noise_scheduler.scale_model_input(latents, t_mid)
                noise_pred = self.unet(latent_in, t_mid, encoder_hidden_states=encoder_hidden_states).sample
                pred_original = self.noise_scheduler.step(noise_pred, t_mid, latents).pred_original_sample
                pred_original = pred_original.to(self.weight_dtype) / self.vae.config.scaling_factor

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    images = self.vae.decode(pred_original).sample.half()

                images = (images / 2 + 0.5).clamp(0, 1)
                images = images.float()
                images = rm_preprocess(images).to('cpu')

                rewards = self.reward_model.score_gard(
                    batch["rm_input_ids"].to('cpu'),
                    batch["rm_attention_mask"].to('cpu'),
                    images,
                )

                hps_scores = torch.tensor(
                    self.hps_model.score(images, batch["caption"]),
                    device=self.device,
                )

                loss = F.relu(-rewards + 2).mean() * args.grad_scale

                avg_reward = rewards.mean().item()
                avg_hps = hps_scores.mean().item()
                logger.info(
                    f"[epoch {epoch + 1} | step {global_step} / {args.max_train_steps}] "
                    f"reward = {avg_reward:.4f}, loss = {loss.item():.4f}  "
                    f"HPS = {avg_hps:+.3f}  "
                )

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                global_step += 1
            #     if global_step >= args.max_train_steps:
            #         break
            # if global_step >= args.max_train_steps:
            #     break

        # save final model
        if args.use_ema:
            self.ema_unet.copy_to(self.unet.parameters())
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info("Training complete. Model saved to %s", args.output_dir)


# ---------------------------------------------
#  CLI entry point
# ---------------------------------------------
if __name__ == "__main__":
    args = get_cfg()

    if torch.cuda.is_available():
        device_name = "cuda"
    elif platform.system() == "Darwin":
        device_name = "mps"
    else:
        device_name = "cpu"

    trainer = Trainer(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        train_parquet="pick-a-pic-v2-unique-prompts/data/train-00000-of-00001.parquet",
        args=args,
        device=device_name,
    )
    trainer.train()
