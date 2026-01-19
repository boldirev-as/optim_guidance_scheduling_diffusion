import torch
import torch.nn.functional as F
from src.trainer.base_trainer import BaseTrainer
from src.constants.dataset import DatasetColumns


class SelfConsistencyTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_min = float(self.cfg_trainer.get("s_min", 0.2))
        self.eps_margin = float(self.cfg_trainer.get("eps_margin", 0.01))
        self.delta_min = float(self.cfg_trainer.get("delta_min", 0.05))

    def _st_indices(self, n, device):
        s_max = 1.0 - self.eps_margin
        s = torch.rand(1, device=device) * (s_max - self.s_min) + self.s_min
        t = torch.clamp(s + self.delta_min, 0.0, 1.0)
        to_idx = lambda x: int((n - 1) * float((1.0 - x).clamp(0, 1)))
        t_idx = to_idx(t)
        s_idx = max(t_idx + 1, to_idx(s))
        s_idx = min(s_idx, n - 1)
        t_idx = min(t_idx, n - 2)
        if s_idx <= t_idx:
            s_idx = min(t_idx + 1, n - 1)
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

    def _sample_image_train(self, batch):
        device = batch[DatasetColumns.tokenized_text.name].device
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=device)
        t_idx, s_idx = self._st_indices(self.cfg_trainer.max_mid_timestep, device)

        with torch.no_grad():
            enc = self.model.get_encoder_hidden_states(
                batch=batch,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            )

        x0 = self._ensure_x0_latents(batch, device)
        a = self.model.noise_scheduler.alphas_cumprod.to(device)
        t_id = int(self.model.timesteps[t_idx])
        s_id = int(self.model.timesteps[s_idx])
        xt = self._q_sample(x0, a[t_id])
        xs = self._q_sample(x0, a[s_id])

        x_s, _ = self.model.do_k_diffusion_steps(
            latents=xt,
            start_timestep_index=t_idx,
            end_timestep_index=s_idx,
            batch=batch,
            encoder_hidden_states=enc,
            return_pred_original=False,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
            seed=batch.get("seeds", [None])[0],
        )

        loss = F.mse_loss(x_s, xs)
        # loss = loss + 0.01 * ((self.model.last_omegas - 7.5) ** 2).mean()
        batch["loss"] = loss

        if self.cfg_trainer.get("requires_score_grad", False):
            batch["image"] = self.model.sample_image(
                latents=xt,
                start_timestep_index=t_idx,
                end_timestep_index=t_idx + 1,
                encoder_hidden_states=enc,
                batch=batch,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                detach_main_path=self.cfg_trainer.detach_main_path,
                seed=batch.get("seeds", [None])[0],
            )

    def _sample_image_eval(self, batch):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = self.cfg_trainer.max_mid_timestep - 1

        with torch.no_grad():
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=None,
                start_timestep_index=0,
                end_timestep_index=mid_timestep,
                batch=batch,
                return_pred_original=False,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                seed=batch.get("seeds", [None])[0]
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=mid_timestep,
            end_timestep_index=mid_timestep + 1,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
            seed=batch.get("seeds", [None])[0]
        )
