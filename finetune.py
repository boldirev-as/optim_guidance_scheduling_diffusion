import logging
import math
import os
import random

from typing import Dict

import wandb
from accelerate import Accelerator
from imscore.hps.model import HPSv2

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed, ProjectConfiguration
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
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
from utils import generate_images, calc_eigvals

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


def collate_images(batch, preprocess):
    # batch - это список dict’ов, каждый содержит ключ "image"
    images = torch.stack([preprocess(sample["image"]) for sample in batch])
    return {"image": images}  # можете вернуть и другие поля, если нужны


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

        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.clip.requires_grad_(False)

        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True, transform_input=False)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval()
        for p in self.inception.parameters():
            p.requires_grad_(False)

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

        if self.accelerator.is_main_process and args.report_to == "wandb":
            import wandb
            wandb.init(
                entity='boldirev-as-life',
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
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
        self.clip.to(self.accelerator.device, dtype=self.weight_dtype)
        self.inception.to('cpu', dtype=self.weight_dtype)

        self.rm_preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.rm_preprocess_eval = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # <-- вот этого не хватало
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        self.ref_eigvals = 0
        self.cur_eigvals = 0

        self.calc_eigenvalues(create_ref_stats=True)

    @torch.no_grad()
    def calc_eigenvalues(self, create_ref_stats=True):
        """extract CLIP‑vision features and cache eigenvalues."""
        logger.info("Computing eigenvalues for SCD/LSCD")

        prompts = open('prompts.txt', 'r').readlines()
        prompts = [prompt.strip() for prompt in prompts]

        seeds = list(range(50))
        self.noise_scheduler.set_timesteps(40, device=self.accelerator.device)

        features = []

        for prompt_idx, prompt in enumerate(prompts):
            print(f'Prompt {prompt_idx}: {prompts[prompt_idx]}')
            for seed in seeds:
                encoded_prompt = self.tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=77
                )
                input_ids = encoded_prompt.input_ids
                generator = torch.Generator(device='cuda').manual_seed(seed)
                images = generate_images(
                    self.unet,
                    self.text_encoder,
                    self.weight_dtype,
                    self.accelerator.device,
                    self.noise_scheduler,
                    self.vae,
                    input_ids,
                    generator=generator,
                )

                image = (images[0] * 2 - 1).clamp(-1, 1)

                image_resized = F.interpolate(
                    image.unsqueeze(0), size=(299, 299),
                    mode='bilinear', align_corners=False
                )

                with torch.no_grad():
                    feat = self.inception(image_resized.cpu())
                    feat = feat.squeeze(0).cpu()
                features.append(feat.detach().clone())

                del images, image, image_resized, feat
                torch.cuda.empty_cache()

        if create_ref_stats:
            self.ref_eigvals = calc_eigvals(features)
        else:
            self.cur_eigvals = calc_eigvals(features)

        logger.info("Eigenvalues ready.")

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

                    mid_timestep = random.randint(36, 39)

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

                    images = (images / 2 + 0.5).clamp_(0, 1)
                    images = self.rm_preprocess(images).to(self.accelerator.device)

                    rewards = self.reward_model.score_gard(
                        batch["rm_input_ids"].to(self.accelerator.device),
                        batch["rm_attention_mask"].to(self.accelerator.device),
                        images,
                    )

                    loss = F.relu(-rewards + 2)

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

                        if self.args.report_to == "wandb":
                            wandb.log({
                                "train/reward": mean_reward,
                                "train/loss": mean_loss,
                                "train/step": global_step
                            }, step=global_step)

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

        for step, batch in enumerate(self.val_dataloader):
            batch_size = batch["input_ids"].shape[0]

            images = generate_images(
                self.unet,
                self.text_encoder,
                self.weight_dtype,
                self.accelerator.device,
                self.noise_scheduler,
                self.vae,
                batch["input_ids"]
            )

            hps_scores = torch.tensor(
                self.hps_model.score(images.to(self.accelerator.device), batch["caption"]),
                device=self.accelerator.device,
            )

            images = self.rm_preprocess(images).to(self.accelerator.device)

            emb = self.clip.vision_model(pixel_values=images).pooler_output
            emb_norm = F.normalize(emb, dim=-1)

            rewards = self.reward_model.score_gard(
                batch["rm_input_ids"].to(self.accelerator.device),
                batch["rm_attention_mask"].to(self.accelerator.device),
                images,
            )

            # ------------- метрика diversity -------------
            sim = emb_norm @ emb_norm.T
            diversity = (1 - sim).triu(1).sum() * 2 / (batch_size * (batch_size - 1))

            # ------------- метрика clip -------------

            img_feat = F.normalize(self.clip.get_image_features(pixel_values=images), dim=-1)  # (B,768)
            text_feat = F.normalize(
                self.clip.get_text_features(input_ids=batch["input_ids"].to(self.accelerator.device)),
                dim=-1)  # (B,768)
            clip_scores = (img_feat * text_feat).sum(-1)

            # ------------- аккумуляция -------------
            running_diversity += diversity
            running_reward += rewards.detach().mean()
            running_hps += hps_scores.detach().mean()
            running_clip += clip_scores.detach().mean()
            accum_count += 1

        running_reward_world = self.accelerator.gather_for_metrics(
            torch.tensor(running_reward, device=self.accelerator.device)
        )
        running_hps_world = self.accelerator.gather_for_metrics(running_hps)
        running_diversity_world = self.accelerator.gather_for_metrics(running_diversity)
        running_clip_world = self.accelerator.gather_for_metrics(running_clip)

        count_world = self.accelerator.gather_for_metrics(
            torch.tensor(accum_count, device=self.accelerator.device)
        ).sum().item()

        if self.accelerator.is_main_process:

            self.calc_eigenvalues(create_ref_stats=False)

            log_ref = torch.log(self.ref_eigvals)
            log_gen = torch.log(self.cur_eigvals)
            diff_log = log_ref - log_gen
            lscd_sum = torch.sum(diff_log ** 2).item()

            running_reward = running_reward_world.sum().item()
            running_hps = running_hps_world.sum().item()
            running_diversity = running_diversity_world.sum().item()
            running_clip = running_clip_world.sum().item()

            logger.info(
                f"[VAL step {global_step}] reward = {running_reward / count_world:.4f}, "
                f"HPS = {running_hps / count_world:.3f}, Diversity = {running_diversity / count_world:.4f}, "
                f"CLIP = {running_clip / count_world:.4f}, "
                f"LSCD = {lscd_sum:.4f}"
            )

            unet_save_path = os.path.join(self.args.output_dir, f"unet_step.pt")
            os.makedirs(self.args.output_dir, exist_ok=True)
            torch.save(self.unet.state_dict(), unet_save_path)
            logger.info(f"Saved UNet weights after step {global_step} to {unet_save_path}")

            if self.args.report_to == "wandb":
                wandb.log({
                    "eval/reward": running_reward / count_world,
                    "eval/HPS": running_hps / count_world,
                    "eval/diversity": running_diversity / count_world,
                    "eval/CLIP": running_clip / count_world,
                    "eval/LSCD": lscd_sum,
                    "eval/step": global_step
                }, step=global_step)

            import torchvision
            grid_img = torchvision.utils.make_grid(images[0].unsqueeze(0).cpu(), normalize=True)
            wandb.log({"eval/sample_image": wandb.Image(grid_img)}, step=global_step)


if __name__ == "__main__":
    args = get_cfg()

    trainer = Trainer(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        train_parquet="pick-a-pic-v2-unique-prompts/data/train-00000-of-00001.parquet",
        args=args
    )
    trainer.train()
