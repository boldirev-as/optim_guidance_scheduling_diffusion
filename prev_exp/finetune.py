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
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel, DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
import ImageReward as RM

from prev_exp.config import get_cfg
from prompt_dataset import ParquetPromptDataset
from utils import generate_images, preprocess_for_inception

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

        if self.accelerator.is_main_process:
            logger.info(self.accelerator.state)

        self.ema_reward = self.args.ema_reward
        self.weight_dtype = torch.float32

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

        self.unet.to(self.accelerator.device)

        self.reward_model = RM.load("ImageReward-v1.0")
        self.reward_model.eval()

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.reward_model.requires_grad_(False)

        if self.accelerator.is_main_process:
            logger.info(
                "Trainable UNet params: %d",
                sum(p.numel() for p in self.unet.parameters() if p.requires_grad),
            )

        if args.report_to == "wandb" and self.accelerator.is_main_process:
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
        if args.max_train_samples:
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

        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler, self.val_dataloader
        )

        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.reward_model.to(self.accelerator.device, dtype=self.weight_dtype)

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
    def calc_eigenvalues(self, create_ref_stats: bool = True):
        """
        Распределённый подсчёт собственных значений ковариационной матрицы Inception-фич.
        Каждый процесс генерит свою часть изображений, копит:
            n_local  = Σ 1
            s_local  = Σ f
            S_local  = Σ f f^T
        Затем делаем all_reduce по всем процессам, считаем ковариацию и eigvals на rank 0,
        рассылаем результат через broadcast.

        Args:
            create_ref_stats: True -> сохраняем в self.ref_eigvals, False -> self.cur_eigvals
        """
        import torch.distributed as dist
        from torchvision.models import inception_v3, Inception_V3_Weights

        device = self.accelerator.device
        rank = self.accelerator.process_index
        world = self.accelerator.num_processes

        distributed = dist.is_available() and dist.is_initialized() and world > 1

        if rank == 0:
            logger.info("Computing eigenvalues for SCD/LSCD … (distributed=%s)", distributed)

        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception.eval().requires_grad_(False).to(device, dtype=self.weight_dtype)

        prompts = [p.strip() for p in open("prompts.txt")]
        seeds = list(range(50))

        tasks = [(seed, start) for seed in seeds for start in range(0, len(prompts), self.args.eval_batch_size)]
        if distributed:
            tasks = tasks[rank::world]

        D = None
        n_local = torch.tensor(0, device=device, dtype=torch.long)
        s_local = None
        S_local = None

        for seed, start in tasks:
            enc = self.tokenizer(
                prompts[start:start + self.args.eval_batch_size],
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=77
            )

            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="scheduler",
                use_karras_sigmas=True
            )
            scheduler.set_timesteps(25, device=device)

            gen = torch.Generator(device=device).manual_seed(seed)

            imgs = generate_images(
                self.unet, self.text_encoder, self.weight_dtype,
                device, scheduler, self.vae, enc.input_ids, generator=gen
            )
            imgs = preprocess_for_inception(imgs, model_type="pytorch_fid")

            feats = inception(imgs.to(device)).to(torch.float64)

            if D is None:
                D = feats.size(1)
                s_local = torch.zeros(D, device=device, dtype=torch.float64)
                S_local = torch.zeros(D, D, device=device, dtype=torch.float64)

            n_b = feats.size(0)
            n_local += n_b
            s_local += feats.sum(dim=0)
            S_local += feats.T @ feats

            del imgs, feats

        if D is None:
            if distributed:
                D_tensor = torch.tensor([0], device=device, dtype=torch.long)
                dist.all_reduce(D_tensor, op=dist.ReduceOp.MAX)
                D_val = D_tensor.item()
                if D_val == 0:
                    raise RuntimeError("Не удалось определить размерность фич (ни у кого не было задач).")
                D = D_val
            else:
                raise RuntimeError("No features computed and not in distributed mode.")
            s_local = torch.zeros(D, device=device, dtype=torch.float64)
            S_local = torch.zeros(D, D, device=device, dtype=torch.float64)

        if distributed:
            dist.all_reduce(n_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(s_local, op=dist.ReduceOp.SUM)
            dist.all_reduce(S_local, op=dist.ReduceOp.SUM)

        if (not distributed) or rank == 0:
            N = n_local.item()
            mean = s_local / N
            cov = (S_local / N) - torch.outer(mean, mean)

            eigvals = torch.linalg.eigvalsh(cov.cpu()).to(self.weight_dtype).to(device)
        else:
            eigvals = torch.empty(D, device=device, dtype=self.weight_dtype)

        # ---------- Рассылка eigvals ----------
        if distributed:
            dist.broadcast(eigvals, src=0)

        eps = 1e-12
        eigvals = torch.clamp(eigvals, min=eps)

        # ---------- Сохранение ----------
        if create_ref_stats:
            self.ref_eigvals = eigvals
        else:
            self.cur_eigvals = eigvals

        if rank == 0:
            logger.info("Eigenvalues ready.")

    def train(self):
        args = self.args
        total_batch_size = args.train_batch_size * args.gradient_accumulation_steps * self.accelerator.num_processes

        if self.accelerator.is_main_process:
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

        if self.accelerator.is_main_process:
            logger.info("Training complete. Model saved to %s", args.output_dir)

    @torch.no_grad()
    def _evaluate(self, global_step: int):
        """
        Вал‑метрики вычисляются на КАЖДОМ процессе (DistributedSampler уже делит датасет).
        После цикла агрегируем суммы/счётчики через Accelerator.gather.
        Логи, сохранение и WandB — только на главном процессе.
        """
        self.unet.eval()

        self.calc_eigenvalues(create_ref_stats=False)

        # --- вспом. модели ---
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip.requires_grad_(False)
        clip.to(self.accelerator.device, dtype=self.weight_dtype)

        hps_model = HPSv2()
        hps_model.requires_grad_(False)
        hps_model.to(self.accelerator.device, dtype=self.weight_dtype)

        # --- счётчики ---
        reward_sum = torch.tensor(0.0, device=self.accelerator.device)
        hps_sum = torch.tensor(0.0, device=self.accelerator.device)
        clip_sum = torch.tensor(0.0, device=self.accelerator.device)
        img_cnt = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)

        diversity_sum = torch.tensor(0.0, device=self.accelerator.device)
        batch_cnt = torch.tensor(0, device=self.accelerator.device, dtype=torch.long)

        # --------------------------------------------------------------
        for batch in self.val_dataloader:
            bsz = batch["input_ids"].size(0)

            scheduler = DPMSolverMultistepScheduler.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="scheduler",
                use_karras_sigmas=True
            )
            scheduler.set_timesteps(25, device=self.accelerator.device)

            # генерация
            images = generate_images(
                self.unet, self.text_encoder, self.weight_dtype,
                self.accelerator.device, scheduler, self.vae, batch["input_ids"]
            )

            # HPS
            hps_scores = torch.tensor(
                hps_model.score(images, batch["caption"]),
                device=self.accelerator.device
            )

            # препроцесс
            images_rm = self.rm_preprocess(images).to(self.accelerator.device)

            # reward
            rewards = self.reward_model.score_gard(
                batch["rm_input_ids"].to(self.accelerator.device),
                batch["rm_attention_mask"].to(self.accelerator.device),
                images_rm,
            )

            # diversity
            emb = clip.vision_model(pixel_values=images_rm).pooler_output
            emb_norm = F.normalize(emb, dim=-1)
            sim = emb_norm @ emb_norm.T
            diversity = (1 - sim).triu(1).sum() * 2 / (bsz * (bsz - 1))

            # CLIP‑similarity
            img_feat = F.normalize(clip.get_image_features(pixel_values=images_rm), dim=-1)
            text_feat = F.normalize(
                clip.get_text_features(input_ids=batch["input_ids"].to(self.accelerator.device)),
                dim=-1
            )
            clip_scores = (img_feat * text_feat).sum(-1)

            # --- накопление ---
            reward_sum += rewards.sum()
            hps_sum += hps_scores.sum()
            clip_sum += clip_scores.sum()
            img_cnt += rewards.numel()

            diversity_sum += diversity
            batch_cnt += 1
        # --------------------------------------------------------------

        # --- агрегация между процессами ---
        reward_sum = self.accelerator.gather(reward_sum).sum()
        hps_sum = self.accelerator.gather(hps_sum).sum()
        clip_sum = self.accelerator.gather(clip_sum).sum()
        img_cnt = self.accelerator.gather(img_cnt).sum()

        diversity_sum = self.accelerator.gather(diversity_sum).sum()
        batch_cnt = self.accelerator.gather(batch_cnt).sum()

        mean_reward = (reward_sum / img_cnt).item()
        mean_hps = (hps_sum / img_cnt).item()
        mean_clip = (clip_sum / img_cnt).item()
        mean_divers = (diversity_sum / batch_cnt).item()

        # --- LSCD (только на главном) ---
        if self.accelerator.is_main_process:
            ref_eigs = torch.sort(self.ref_eigvals, descending=True).values
            cur_eigs = torch.sort(self.cur_eigvals, descending=True).values
            lscd_sum = torch.sum((torch.log(ref_eigs) - torch.log(cur_eigs)) ** 2).item()

            logger.info(
                f"[VAL step {global_step}] "
                f"reward={mean_reward:.4f}  HPS={mean_hps:.3f}  "
                f"Diversity={mean_divers:.4f}  CLIP={mean_clip:.4f}  LSCD={lscd_sum:.4f}"
            )

            # сохранение модели
            os.makedirs(self.args.output_dir, exist_ok=True)
            unet_path = os.path.join(self.args.output_dir, f"unet_step.pt")
            torch.save(self.accelerator.unwrap_model(self.unet).state_dict(), unet_path)

            # WandB
            if self.args.report_to == "wandb":
                wandb.log({
                    "eval/reward": mean_reward,
                    "eval/HPS": mean_hps,
                    "eval/divers": mean_divers,
                    "eval/CLIP": mean_clip,
                    "eval/LSCD": lscd_sum,
                    "eval/step": global_step
                }, step=global_step)

            # пример изображения
            import torchvision
            grid_img = torchvision.utils.make_grid(images[0].unsqueeze(0).cpu(), normalize=True)
            wandb.log({"eval/sample_image": wandb.Image(grid_img)}, step=global_step)

        self.accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = get_cfg()

    trainer = Trainer(
        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
        train_parquet="pick-a-pic-v2-unique-prompts/data/train-00000-of-00001.parquet",
        args=args
    )
    trainer.train()
