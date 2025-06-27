import logging
import math
import os
import random

from typing import Dict

from accelerate import Accelerator
from datasets import load_dataset
from imscore.hps.model import HPSv2

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed, ProjectConfiguration
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import ImageReward as RM

from config import get_cfg
from prompt_dataset import ParquetPromptDataset

BICUBIC = InterpolationMode.BICUBIC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("refl_trainer")


def collate_fn(batch) -> Dict[str, torch.Tensor]:
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

        self.ema_reward = self.args.ema_reward

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

        self.clip_vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_vision.requires_grad_(False)
        self.clip_text = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_text.requires_grad_(False)

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

        self.val_dataset = ParquetPromptDataset(
            parquet_path=self.args.validation_json,
            tokenizer=self.tokenizer,
            rm_tokenizer=self.reward_model.blip.tokenizer,
            train_mode=False,
            seed=args.seed or 42,
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

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
        self.clip_vision.to(self.accelerator.device, dtype=self.weight_dtype)
        self.clip_text.to(self.accelerator.device, dtype=self.weight_dtype)

        self.rm_preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        self._build_ref_stats()

    def _build_ref_stats(self):
        """Load HPDv2‑test, extract CLIP‑vision features and cache eigenvalues."""
        logger.info("Computing reference eigenvalues for SCD/LSCD (HPDv2‑test)…")
        ds = load_dataset("ymhao/HPDv2", split="test")
        ref_loader = DataLoader(
            ds,
            batch_size=self.args.eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            shuffle=False,
        )

        feats = []
        for batch in ref_loader:
            imgs = self.rm_preprocess(batch["image"]).to(self.accelerator.device)
            with torch.no_grad():
                f = self.clip_vision(imgs).pooler_output  # (B, D=768)
            feats.append(f)

        feats = torch.cat(feats, dim=0)  # (N, D)
        mu = feats.mean(dim=0)
        # unbiased sample covariance
        Sigma = (feats - mu).T @ (feats - mu) / (feats.size(0) - 1)
        eigvals, _ = torch.linalg.eigh(Sigma)
        self.ref_eig = eigvals.clamp_min_(1e-9).detach()
        logger.info("Reference eigenvalues ready (D=%d).", eigvals.size(0))

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

        for epoch in range(args.num_train_epochs):
            self.unet.train()
            self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)

            running_reward = 0.0
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
                    images = self.rm_preprocess(images).to(self.accelerator.device)
                    images.requires_grad_(True)

                    rewards = self.reward_model.score_gard(
                        batch["rm_input_ids"].to(self.accelerator.device),
                        batch["rm_attention_mask"].to(self.accelerator.device),
                        images,
                    )

                    self.ema_reward = 0.9 * self.ema_reward + 0.1 * rewards.mean().item()
                    loss = -(rewards - self.ema_reward).mean()

                    running_reward += rewards.detach().mean()
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
                    mean_loss = running_loss / accum_count

                    mean_reward = self.accelerator.gather(mean_reward).mean().item()
                    mean_loss = self.accelerator.gather(mean_loss).mean().item()

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"[epoch {epoch + 1} | step {global_step} / {args.max_train_steps}] "
                            f"reward = {mean_reward:.4f}, loss = {mean_loss:.4f}  "
                        )

                    running_reward = 0.0
                    accum_count = 0
                    running_loss = 0.0

                    global_step += 1

                    if global_step % args.evaluation_steps == 0:
                        self._evaluate(global_step)

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

    @torch.no_grad()
    def _evaluate(self, global_step: int):

        self.unet.eval()
        self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)

        running_reward = 0.0
        running_hps = 0.0
        accum_count = 0
        running_diversity = 0.0
        running_clip = 0.0

        gen_feats = []

        for step, batch in enumerate(self.val_dataloader):
            batch_size = batch["input_ids"].shape[0]

            encoder_hidden_states = self.text_encoder(batch["input_ids"].to(self.accelerator.device))[0]
            latents = torch.randn((batch_size, 4, 64, 64), device=self.accelerator.device,
                                  dtype=self.weight_dtype)

            for t in self.noise_scheduler.timesteps:
                latent_in = self.noise_scheduler.scale_model_input(latents, t)
                noise_pred = self.unet(latent_in, t, encoder_hidden_states=encoder_hidden_states).sample
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

            pred_original = latents.float() / self.vae.config.scaling_factor
            images = self.vae.decode(pred_original).sample
            images = (images / 2 + 0.5).clamp(0, 1)

            hps_scores = torch.tensor(
                self.hps_model.score(images.to(self.accelerator.device), batch["caption"]),
                device=self.accelerator.device,
            )

            images = self.rm_preprocess(images).to(self.accelerator.device)
            with torch.no_grad():
                emb = self.clip_vision(images).pooler_output  # (B, D)
            emb_norm = F.normalize(emb, dim=-1)
            gen_feats.append(emb)

            rewards = self.reward_model.score_gard(
                batch["rm_input_ids"].to(self.accelerator.device),
                batch["rm_attention_mask"].to(self.accelerator.device),
                images,
            )

            # ------------- метрика diversity -------------
            sim = emb_norm @ emb_norm.T
            diversity = (1 - sim).triu(1).sum() * 2 / (batch_size * (batch_size - 1))
            txt_emb = F.normalize(
                self.clip_text(batch["input_ids"].to(self.accelerator.device))[0][:, 0, :],
                dim=-1,
            )
            clip_scores = (txt_emb * emb_norm).sum(dim=-1)

            # ------------- аккумуляция -------------
            running_diversity += diversity.item()
            running_reward += rewards.detach().mean()
            running_hps += hps_scores.detach().mean()
            running_clip += clip_scores.detach().mean()
            accum_count += 1

        gen_feats = torch.cat(gen_feats, dim=0)  # (M, D)
        mu_g = gen_feats.mean(dim=0)
        Sigma_g = (gen_feats - mu_g).T @ (gen_feats - mu_g) / (gen_feats.size(0) - 1)
        eig_g, _ = torch.linalg.eigh(Sigma_g)
        eig_g = eig_g.clamp_min_(1e-9)

        lscd = torch.sum((torch.log(self.ref_eig) - torch.log(eig_g)) ** 2).item()

        if self.accelerator.is_main_process:
            logger.info(
                f"[VAL step {global_step}] reward = {running_reward / accum_count:.4f}, "
                f"HPS = {running_hps / accum_count:.3f}, Diversity = {running_diversity / accum_count:.4f}, "
                f"CLIP = {running_clip / accum_count:.4f}, "
                f"LSCD = {lscd:.4f}"
            )


if __name__ == "__main__":
    args = get_cfg()

    trainer = Trainer(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        train_parquet="pick-a-pic-v2-unique-prompts/data/train-00000-of-00001.parquet",
        args=args
    )
    trainer.train()
