import torch
from src.trainer.base_trainer import BaseTrainer
from src.constants.dataset import DatasetColumns


class SelfConsistencyTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_min = float(self.cfg_trainer.get("s_min", 0.01))
        self.zeta = float(self.cfg_trainer.get("zeta", self.cfg_trainer.get("eps_margin", 1e-2)))
        self.delta = float(self.cfg_trainer.get("delta", self.cfg_trainer.get("delta_min", 0.2)))
        self.num_steps = int(self.cfg_trainer.get("num_steps", 64))
        self.num_particles = int(self.cfg_trainer.get("m", 4))
        self.beta_sc = float(self.cfg_trainer.get("beta", 2.0))
        self.lambda_sc = float(self.cfg_trainer.get("lambda_sc", 0.0))

    def _st_indices(self, n, batch_size, device):
        s_max = 1.0 - self.zeta - self.delta
        if s_max <= self.s_min:
            s_max = self.s_min + 1e-6
        s = torch.rand(batch_size, device=device) * (s_max - self.s_min) + self.s_min
        delta_max = 1.0 - self.zeta - s
        delta_max = torch.where(
            delta_max <= self.delta,
            torch.full_like(delta_max, self.delta + 1e-6),
            delta_max,
        )
        delta_t = torch.rand(batch_size, device=device) * (delta_max - self.delta) + self.delta
        t = s + delta_t

        t_idx = ((1.0 - t).clamp(0, 1) * (n - 1)).long()
        s_idx = ((1.0 - s).clamp(0, 1) * (n - 1)).long()
        s_idx = torch.maximum(s_idx, t_idx + 1)
        s_idx = torch.clamp(s_idx, max=n - 1)
        t_idx = torch.clamp(t_idx, max=n - 2)
        return t_idx, s_idx

    @staticmethod
    def _q_sample(x0, a_bar):
        a = a_bar.sqrt()
        s = (1.0 - a_bar).sqrt()
        return a * x0 + s * torch.randn_like(x0)

    def _ensure_x0_latents(self, batch, device):
        img = batch[DatasetColumns.original_image.name]
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            enc_out = self.model.vae.encode(img)
            x0 = enc_out.latent_dist.sample()
        x0 = x0 * self.model.vae.config.scaling_factor
        batch["x0_latents"] = x0
        return x0

    def _slice_batch(self, batch, idx):
        sliced = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.size(0) == batch[DatasetColumns.tokenized_text.name].size(0):
                sliced[key] = value[idx:idx + 1]
            elif isinstance(value, list) and len(value) == batch[DatasetColumns.tokenized_text.name].size(0):
                sliced[key] = [value[idx]]
            else:
                sliced[key] = value
        return sliced

    def _repeat_batch(self, batch, repeats):
        repeated = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.size(0) == 1:
                repeated[key] = value.repeat(repeats, *[1] * (value.dim() - 1))
            elif isinstance(value, list) and len(value) == 1:
                repeated[key] = value * repeats
            else:
                repeated[key] = value
        return repeated

    def _sample_image_train(self, batch):
        device = batch[DatasetColumns.tokenized_text.name].device
        batch_size = batch[DatasetColumns.tokenized_text.name].size(0)
        self.model.set_timesteps(self.num_steps, device=device)
        t_idx, s_idx = self._st_indices(self.num_steps, batch_size, device)

        x0 = self._ensure_x0_latents(batch, device)
        a = self.model.noise_scheduler.alphas_cumprod.to(device)
        losses = []
        reward_images = []

        for i in range(batch_size):
            t_id = int(self.model.timesteps[t_idx[i]])
            s_id = int(self.model.timesteps[s_idx[i]])
            x0_i = x0[i:i + 1]

            eps_t = torch.randn(
                (self.num_particles,) + x0_i.shape[1:], device=device, dtype=x0_i.dtype
            )
            eps_s = torch.randn_like(eps_t)
            xt = a[t_id].sqrt() * x0_i + (1.0 - a[t_id]).sqrt() * eps_t
            xs_true = a[s_id].sqrt() * x0_i + (1.0 - a[s_id]).sqrt() * eps_s

            batch_i = self._slice_batch(batch, i)
            batch_m = self._repeat_batch(batch_i, self.num_particles)
            with torch.no_grad():
                enc = self.model.get_encoder_hidden_states(
                    batch=batch_m,
                    do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                )

            x_s_tilde, _ = self.model.do_k_diffusion_steps(
                latents=xt,
                start_timestep_index=int(t_idx[i]),
                end_timestep_index=int(s_idx[i]),
                batch=batch_m,
                encoder_hidden_states=enc,
                return_pred_original=False,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                detach_main_path=self.cfg_trainer.detach_main_path,
                seed=batch.get("seeds", [None])[0],
            )

            diff = (x_s_tilde - xs_true).float()
            l2 = diff.flatten(1).pow(2).sum(1).sqrt()
            term1 = l2.pow(self.beta_sc).mean()

            if self.num_particles > 1 and self.lambda_sc > 0:
                flat = x_s_tilde.float().flatten(1)
                dists = torch.cdist(flat, flat, p=2)
                pairwise = dists.pow(self.beta_sc).sum() / (self.num_particles * (self.num_particles - 1))
                loss_i = term1 - 0.5 * self.lambda_sc * pairwise
            else:
                loss_i = term1

            losses.append(loss_i)

            if self.cfg_trainer.get("requires_score_grad", False):
                pred_original_sample = x_s_tilde[:1] / self.model.vae.config.scaling_factor
                raw_image = self.model.vae.decode(pred_original_sample).sample
                reward_images.append(self.model.get_reward_image(raw_image))

        batch["loss"] = torch.stack(losses).mean()

        if self.cfg_trainer.get("requires_score_grad", False):
            batch["image"] = torch.cat(reward_images, dim=0)

    def _sample_image_eval(self, batch):
        self.model.set_timesteps(self.num_steps, device=self.device)
        batch["image"] = self.model.sample_image(
            latents=None,
            start_timestep_index=0,
            end_timestep_index=self.num_steps,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
            seed=batch.get("seeds", [None])[0],
        )
