import torch
import torch.nn.functional as F

from src.constants.dataset import DatasetColumns
from src.trainer.guidance_trainer import SelfConsistencyTrainer


class RCSGPTrainer(SelfConsistencyTrainer):
    """
    Self-consistency trainer with a stochastic guidance policy (RC-SGP):
    - actor/value-style policy shaping from self-consistency signal
    - risk-constrained optimization via dual variable lambda
    - smoothness + KL regularization for omega schedules
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.actor_loss_scale = float(self.cfg_trainer.get("actor_loss_scale", 0.05))
        self.value_loss_scale = float(self.cfg_trainer.get("value_loss_scale", 0.1))
        self.risk_loss_scale = float(self.cfg_trainer.get("risk_loss_scale", 0.2))
        self.kl_loss_scale = float(self.cfg_trainer.get("kl_loss_scale", 0.01))
        self.smooth_loss_scale = float(self.cfg_trainer.get("smooth_loss_scale", 0.02))
        self.entropy_loss_scale = float(self.cfg_trainer.get("entropy_loss_scale", 0.001))

        self.risk_alpha = float(self.cfg_trainer.get("risk_alpha", 0.15))
        self.risk_sc_threshold = float(self.cfg_trainer.get("risk_sc_threshold", 1.5))
        self.risk_sc_margin = float(self.cfg_trainer.get("risk_sc_margin", 0.05))
        self.risk_dual_lr = float(self.cfg_trainer.get("risk_dual_lr", 0.01))
        self.lambda_dual = float(self.cfg_trainer.get("risk_lambda_init", 0.0))

        self.sc_ema_decay = float(self.cfg_trainer.get("sc_ema_decay", 0.95))
        self.sc_ema: float | None = None

    def _get_train_loss_names(self):
        names = super()._get_train_loss_names()
        names.extend(
            [
                "sc_loss",
                "policy_actor_loss",
                "policy_value_loss",
                "policy_risk_bce",
                "policy_kl_loss",
                "policy_smooth_loss",
                "policy_entropy",
                "policy_constraint",
                "policy_loss",
                "policy_omega_mean",
                "policy_omega_std",
                "policy_lambda",
            ]
        )
        return names

    def _get_policy_step_infos(self):
        if self.model.guidance_net is None:
            return []
        if hasattr(self.model.guidance_net, "pop_step_infos"):
            return self.model.guidance_net.pop_step_infos()
        return []

    def _get_zero(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    def _update_sc_ema(self, value: float) -> None:
        if self.sc_ema is None:
            self.sc_ema = value
            return
        self.sc_ema = self.sc_ema_decay * self.sc_ema + (1.0 - self.sc_ema_decay) * value

    def _policy_losses(
            self,
            step_infos: list[dict[str, torch.Tensor]],
            sc_term: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = self._get_zero(sc_term)
        if not step_infos:
            return {
                "policy_actor_loss": zero,
                "policy_value_loss": zero,
                "policy_risk_bce": zero,
                "policy_kl_loss": zero,
                "policy_smooth_loss": zero,
                "policy_entropy": zero,
                "policy_constraint": zero,
                "policy_loss": zero,
                "policy_omega_mean": zero,
                "policy_omega_std": zero,
                "policy_lambda": torch.tensor(self.lambda_dual, device=sc_term.device, dtype=sc_term.dtype),
            }

        log_prob = torch.stack([x["log_prob"].reshape(-1).mean() for x in step_infos]).mean()
        entropy = torch.stack([x["entropy"].reshape(-1).mean() for x in step_infos]).mean()

        omega_series = torch.stack([x["omega"].reshape(-1).mean() for x in step_infos])
        if omega_series.numel() > 1:
            smooth_loss = (omega_series[1:] - omega_series[:-1]).abs().mean()
            omega_std = omega_series.std(unbiased=False)
        else:
            smooth_loss = zero
            omega_std = zero
        omega_mean = omega_series.mean()

        kl_terms = []
        for x in step_infos:
            if all(k in x for k in ("mu", "log_std", "prior_mu", "prior_log_std")):
                mu = x["mu"]
                log_std = x["log_std"]
                prior_mu = x["prior_mu"]
                prior_log_std = x["prior_log_std"]
                var = torch.exp(2.0 * log_std)
                prior_var = torch.exp(2.0 * prior_log_std)
                kl = prior_log_std - log_std + (var + (mu - prior_mu).pow(2)) / (2.0 * prior_var) - 0.5
                kl_terms.append(kl.mean())
        kl_loss = torch.stack(kl_terms).mean() if kl_terms else zero

        sc_value = float(sc_term.detach().item())
        ema_ref = self.sc_ema if self.sc_ema is not None else sc_value
        bad_threshold = max(self.risk_sc_threshold, ema_ref + self.risk_sc_margin)
        bad_event = 1.0 if sc_value > bad_threshold else 0.0

        risk_logits = torch.cat([x["risk_logit"].reshape(-1) for x in step_infos], dim=0)
        risk_target = torch.full_like(risk_logits, fill_value=bad_event)
        risk_bce = F.binary_cross_entropy_with_logits(risk_logits, risk_target)
        risk_probs = torch.sigmoid(risk_logits)
        constraint = risk_probs.mean() - self.risk_alpha

        value_loss = zero
        if all("value" in x for x in step_infos):
            value_pred = torch.cat([x["value"].reshape(-1) for x in step_infos], dim=0)
            value_target = torch.full_like(value_pred, fill_value=-sc_value)
            value_loss = F.smooth_l1_loss(value_pred, value_target)

        adv = torch.tensor(ema_ref - sc_value, device=sc_term.device, dtype=sc_term.dtype)
        actor_loss = -(adv * log_prob)

        self.lambda_dual = max(0.0, self.lambda_dual + self.risk_dual_lr * float(constraint.detach().item()))
        lambda_tensor = torch.tensor(self.lambda_dual, device=sc_term.device, dtype=sc_term.dtype)
        policy_loss = (
                self.actor_loss_scale * actor_loss
                + self.value_loss_scale * value_loss
                + self.risk_loss_scale * risk_bce
                + self.kl_loss_scale * kl_loss
                + self.smooth_loss_scale * smooth_loss
                - self.entropy_loss_scale * entropy
                + lambda_tensor * constraint
        )

        self._update_sc_ema(sc_value)

        return {
            "policy_actor_loss": actor_loss,
            "policy_value_loss": value_loss,
            "policy_risk_bce": risk_bce,
            "policy_kl_loss": kl_loss,
            "policy_smooth_loss": smooth_loss,
            "policy_entropy": entropy,
            "policy_constraint": constraint,
            "policy_loss": policy_loss,
            "policy_omega_mean": omega_mean,
            "policy_omega_std": omega_std,
            "policy_lambda": lambda_tensor,
        }

    def _sample_image_train(self, batch):
        device = batch[DatasetColumns.tokenized_text.name].device
        batch_size = batch[DatasetColumns.tokenized_text.name].size(0)
        self.model.set_timesteps(self.num_steps, device=device)
        t_idx, s_idx = self._st_indices(self.num_steps, batch_size, device)

        x0 = self._ensure_x0_latents(batch, device)
        a = self.model.noise_scheduler.alphas_cumprod.to(device)
        losses = []
        reward_images = []

        log_accumulator: dict[str, list[torch.Tensor]] = {
            "sc_loss": [],
            "policy_actor_loss": [],
            "policy_value_loss": [],
            "policy_risk_bce": [],
            "policy_kl_loss": [],
            "policy_smooth_loss": [],
            "policy_entropy": [],
            "policy_constraint": [],
            "policy_loss": [],
            "policy_omega_mean": [],
            "policy_omega_std": [],
            "policy_lambda": [],
        }

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

            step_infos = self._get_policy_step_infos()

            diff = (x_s_tilde - xs_true).float()
            l2 = diff.flatten(1).pow(2).sum(1).sqrt()
            term1 = l2.pow(self.beta_sc).mean()
            log_accumulator["sc_loss"].append(term1.detach())

            if self.num_particles > 1 and self.lambda_sc > 0:
                flat = x_s_tilde.float().flatten(1)
                dists = torch.cdist(flat, flat, p=2)
                pairwise = dists.pow(self.beta_sc).sum() / (self.num_particles * (self.num_particles - 1))
                sc_loss = term1 - 0.5 * self.lambda_sc * pairwise
            else:
                sc_loss = term1

            policy_stats = self._policy_losses(step_infos=step_infos, sc_term=term1)
            for key, value in policy_stats.items():
                if key in log_accumulator:
                    log_accumulator[key].append(value.detach())

            loss_i = sc_loss + policy_stats["policy_loss"]
            losses.append(loss_i)

            if self.cfg_trainer.get("requires_score_grad", False):
                pred_original_sample = x_s_tilde[:1] / self.model.vae.config.scaling_factor
                raw_image = self.model.vae.decode(pred_original_sample).sample
                reward_images.append(self.model.get_reward_image(raw_image))

        batch["loss"] = torch.stack(losses).mean()
        for key, values in log_accumulator.items():
            if values:
                batch[key] = torch.stack(values).mean()

        if self.cfg_trainer.get("requires_score_grad", False):
            batch["image"] = torch.cat(reward_images, dim=0)
