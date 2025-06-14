import logging
import math
import os
import platform
import random

import diffusers

from typing import Dict, Any

import transformers
from accelerate import Accelerator
from imscore.hps.model import HPSv2

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate.utils import set_seed, ProjectConfiguration
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
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


# -----------------------------------------------------------------------------
# 1) Structured schema (unchanged)
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # --- core ---
    grad_scale: float = 1
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
    max_train_steps: Optional[int] = 10000
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    base_lr: float = 2.5e-6
    learning_rate: float = base_lr * train_batch_size * gradient_accumulation_steps
    scale_lr: bool = False
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    snr_gamma: Optional[float] = None
    use_8bit_adam: bool = False

    # --- optimiser ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5

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
    dataloader_num_workers: int = 8


# -----------------------------------------------------------------------------
# 2) Helper replacing `parse_args` (unchanged apart from local_rank env)
# -----------------------------------------------------------------------------

def get_cfg() -> DictConfig:
    """Merge defaults, YAML file (if any) and CLI overrides."""
    cfg = OmegaConf.structured(TrainConfig)

    cli = OmegaConf.from_cli()
    if "config" in cli:
        yaml_cfg = OmegaConf.load(cli.pop("config"))
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    cfg = OmegaConf.merge(cfg, cli)

    cfg.non_ema_revision = cfg.revision

    return cfg


# -----------------------------------------------------------------------------
# Datasets & Collate (unchanged)
# -----------------------------------------------------------------------------
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
            "caption": caption,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    out = {}
    for k in batch[0]:
        if k == "caption":
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out


class Trainer:
    def __init__(self, pretrained_model_name_or_path: str, train_parquet: str, args):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.args = args
        self.weight_dtype = torch.float32

        accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.report_to,
            project_config=accelerator_project_config
        )

        if args.seed is not None:
            set_seed(args.seed)
            self.accelerator.state.initialize_random_seed(args.seed)

        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)

        logger.info(self.accelerator.state)

        # sched / tokeniser / models
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae", revision=args.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
        )

        self.reward_model = RM.load("ImageReward-v1.0")
        self.reward_model.eval()
        self.hps_model = HPSv2()
        self.hps_model.requires_grad_(False)

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.reward_model.requires_grad_(False)

        logger.info(
            "Trainable UNet params: %d",
            sum(p.numel() for p in self.unet.parameters() if p.requires_grad),
        )

        # memory‑efficient attention
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise RuntimeError("xformers not available, but requested.")

        if args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

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
        if args.max_train_samples and self.accelerator.is_main_process:
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
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.reward_model.to(self.accelerator.device, dtype=self.weight_dtype)
        self.hps_model.to(self.accelerator.device, dtype=self.weight_dtype)

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train(self):
        args = self.args
        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * self.accelerator.num_processes

        logger.info("***** Running training *****")
        logger.info("Num examples           = %d", len(self.train_dataset))
        logger.info("Num epochs             = %d", args.num_train_epochs)
        logger.info("Instant batch size     = %d", args.train_batch_size)
        logger.info("Total batch size       = %d", total_batch_size)
        logger.info("Gradient accum steps   = %d", args.gradient_accumulation_steps)
        logger.info("Total optimisation steps = %d", args.max_train_steps)
        logger.info("Num processes          = %f", self.accelerator.num_processes)

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
            self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)

            running_reward = 0.0
            running_hps = 0.0
            accum_count = 0
            running_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                batch_size = batch["input_ids"].shape[0]

                with self.accelerator.accumulate(self.unet):

                    encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device))[0]

                    # random latents
                    latents = torch.randn((batch_size, 4, 64, 64), device=self.accelerator.device,
                                          dtype=self.weight_dtype)

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
                    pred_original = pred_original.float() / self.vae.config.scaling_factor

                    images = self.vae.decode(pred_original).sample

                    images = (images / 2 + 0.5).clamp(0, 1)
                    images = rm_preprocess(images).to(self.accelerator.device)
                    images = images.detach().requires_grad_(True)

                    rewards = self.reward_model.score_gard(
                        batch["rm_input_ids"].to(self.accelerator.device),
                        batch["rm_attention_mask"].to(self.accelerator.device),
                        images,
                    )

                    hps_scores = torch.tensor(
                        self.hps_model.score(images.to(self.accelerator.device), batch["caption"]),
                        device=self.accelerator.device,
                    )

                    loss = -rewards.mean()

                    running_reward += rewards.detach().mean()
                    running_hps += hps_scores.detach().mean()
                    running_loss += loss.detach().mean()
                    accum_count += 1

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:

                    mean_reward = running_reward / accum_count
                    mean_hps = running_hps / accum_count
                    mean_loss = running_loss / accum_count

                    mean_reward = self.accelerator.gather(mean_reward).mean().item()
                    mean_hps = self.accelerator.gather(mean_hps).mean().item()
                    mean_loss = self.accelerator.gather(mean_loss).mean().item()

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"[epoch {epoch + 1} | step {global_step} / {args.max_train_steps}] "
                            f"reward = {mean_reward:.4f}, loss = {mean_loss:.4f}  "
                            f"HPS = {mean_hps:.3f}  "
                        )

                    running_reward = 0.0
                    running_hps = 0.0
                    accum_count = 0
                    running_loss = 0.0

                    global_step += 1

                if global_step >= args.max_train_steps:
                    break
            if global_step >= args.max_train_steps:
                break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

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

    trainer = Trainer(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        train_parquet="pick-a-pic-v2-unique-prompts/data/train-00000-of-00001.parquet",
        args=args
    )
    trainer.train()
