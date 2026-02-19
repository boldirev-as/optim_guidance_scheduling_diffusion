import math

import torch
import torch.nn.functional as F
from torch import nn


class RCSGPGuidanceNet(nn.Module):
    """
    Risk-Constrained Stochastic Guidance Policy (RC-SGP).

    Produces timestep-wise guidance weight omega_t in [omega_min, omega_max],
    and stores step-wise policy statistics for trainer-side regularization:
    log_prob, entropy surrogate, risk logit, value baseline, and KL terms.
    """

    def __init__(
            self,
            cond_dim: int,
            time_emb_dim: int,
            hidden_dim: int = 1024,
            num_mixtures: int = 3,
            base_scale: float = 7.5,
            omega_min: float = 0.0,
            omega_max: float = 12.0,
            prior_std: float = 0.75,
            prior_start: float = 7.5,
            prior_end: float = 5.5,
            stochastic_train: bool = True,
            stochastic_eval: bool = False,
            pad_id: int | None = None,
            bos_id: int | None = None,
            use_attn_pool: bool = True,
    ):
        super().__init__()
        if num_mixtures < 1:
            raise ValueError("num_mixtures must be >= 1")
        if omega_max <= omega_min:
            raise ValueError("omega_max must be > omega_min")
        if prior_std <= 0:
            raise ValueError("prior_std must be > 0")

        self.num_mixtures = int(num_mixtures)
        self.base_scale = float(base_scale)
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.prior_std = float(prior_std)
        self.prior_start = float(prior_start)
        self.prior_end = float(prior_end)
        self.stochastic_train = bool(stochastic_train)
        self.stochastic_eval = bool(stochastic_eval)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.use_attn_pool = use_attn_pool
        self._eps = 1e-6

        self.pool_lin = nn.Linear(cond_dim, cond_dim)
        self.pool_v = nn.Linear(cond_dim, 1, bias=False)

        # state features: ||delta_eps||, cos(eps_c, eps_u), ||x_t||, t_norm, prev_omega_norm
        state_dim = 5
        self.backbone = nn.Sequential(
            nn.Linear(cond_dim + time_emb_dim + state_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        self.mix_logits = nn.Linear(hidden_dim, self.num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, self.num_mixtures)
        self.log_std_head = nn.Linear(hidden_dim, self.num_mixtures)

        self.risk_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._base_z = self._omega_to_logit(
            torch.tensor([self.base_scale], dtype=torch.float32)
        ).item()
        nn.init.constant_(self.mu_head.bias, self._base_z)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.log_std_head.bias, math.log(0.35))
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.zeros_(self.mix_logits.weight)
        nn.init.zeros_(self.mix_logits.bias)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

        self._step_infos: list[dict[str, torch.Tensor]] = []

    def reset_step_infos(self) -> None:
        self._step_infos = []

    def pop_step_infos(self) -> list[dict[str, torch.Tensor]]:
        infos = self._step_infos
        self._step_infos = []
        return infos

    def _make_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        if self.pad_id is not None:
            mask &= (input_ids != self.pad_id)
        if self.bos_id is not None:
            mask &= (input_ids != self.bos_id)
        return mask

    def _pool_tokens(self, hidden_cond: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_attn_pool:
            return hidden_cond.mean(dim=1)
        mask = self._make_mask(input_ids)
        h = torch.tanh(self.pool_lin(hidden_cond))
        scores = self.pool_v(h).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (hidden_cond * weights.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def _time_embed(unet, sample, timestep, timestep_cond=None, encoder_hidden_states=None, added_cond_kwargs=None):
        t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
        emb = unet.time_embedding(t_emb, timestep_cond)
        aug_emb = unet.get_aug_embed(
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        if unet.config.addition_embed_type == "image_hint":
            aug_emb, _ = aug_emb
        emb = emb + aug_emb if aug_emb is not None else emb
        if unet.time_embed_act is not None:
            emb = unet.time_embed_act(emb)
        return emb

    def _omega_to_logit(self, omega: torch.Tensor) -> torch.Tensor:
        scale = self.omega_max - self.omega_min
        u = (omega - self.omega_min) / scale
        u = u.clamp(self._eps, 1.0 - self._eps)
        return torch.logit(u)

    def _prior_mean_z(self, t_norm: torch.Tensor) -> torch.Tensor:
        prior_omega = self.prior_end + (self.prior_start - self.prior_end) * t_norm
        prior_omega = prior_omega.clamp(self.omega_min + 1e-3, self.omega_max - 1e-3)
        return self._omega_to_logit(prior_omega)

    @staticmethod
    def _gaussian_log_prob(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-2.0 * log_std)
        return -0.5 * ((z - mu) ** 2 * inv_var + 2.0 * log_std + math.log(2.0 * math.pi))

    @staticmethod
    def _gaussian_entropy(log_std: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std

    def _extract_state_features(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            noise_pred_uncond: torch.Tensor | None,
            noise_pred_text: torch.Tensor | None,
            prev_omega: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = sample.size(0)
        device = sample.device
        dtype = sample.dtype

        sample_flat = sample.float().flatten(1)
        sample_norm = sample_flat.norm(dim=1, keepdim=True)

        if noise_pred_uncond is not None and noise_pred_text is not None:
            delta = (noise_pred_text - noise_pred_uncond).float().flatten(1)
            delta_norm = delta.norm(dim=1, keepdim=True)
            cos_sim = F.cosine_similarity(
                noise_pred_text.float().flatten(1),
                noise_pred_uncond.float().flatten(1),
                dim=1,
                eps=1e-8,
            ).unsqueeze(1)
        else:
            delta_norm = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)
            cos_sim = torch.zeros((batch_size, 1), device=device, dtype=torch.float32)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([float(timestep)], device=device)
        timestep = timestep.to(device=device, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        t_norm = (timestep / 999.0).unsqueeze(1).clamp(0.0, 1.0)

        if prev_omega is None:
            prev_omega = torch.full((batch_size, 1), self.base_scale, device=device, dtype=torch.float32)
        else:
            prev_omega = prev_omega.reshape(batch_size, -1)[:, :1].float()
        prev_omega_norm = (prev_omega - self.omega_min) / (self.omega_max - self.omega_min)

        state = torch.cat(
            [
                delta_norm,
                cos_sim,
                sample_norm,
                t_norm,
                prev_omega_norm,
            ],
            dim=1,
        )
        return state.to(dtype=dtype)

    def forward(
            self,
            *,
            unet,
            encoder_hidden_states: torch.Tensor,
            input_ids: torch.Tensor,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            timestep_cond: torch.Tensor | None = None,
            added_cond_kwargs: dict | None = None,
            noise_pred_uncond: torch.Tensor | None = None,
            noise_pred_text: torch.Tensor | None = None,
            prev_omega: torch.Tensor | None = None,
            timestep_index: int | None = None,
            **_,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        if encoder_hidden_states.size(0) == 2 * batch_size:
            _, hidden_cond = encoder_hidden_states.chunk(2)
        else:
            hidden_cond = encoder_hidden_states

        cond_embed = self._pool_tokens(hidden_cond, input_ids)
        emb = self._time_embed(
            unet=unet,
            sample=sample,
            timestep=timestep,
            timestep_cond=timestep_cond,
            encoder_hidden_states=hidden_cond,
            added_cond_kwargs=added_cond_kwargs,
        )

        state = self._extract_state_features(
            sample=sample,
            timestep=timestep,
            noise_pred_uncond=noise_pred_uncond,
            noise_pred_text=noise_pred_text,
            prev_omega=prev_omega,
        )
        h = self.backbone(torch.cat([cond_embed, emb, state], dim=-1))

        mix_logits = self.mix_logits(h)
        mix_log_probs = torch.log_softmax(mix_logits, dim=-1)
        mu_all = self.mu_head(h)
        log_std_all = self.log_std_head(h).clamp(min=-4.0, max=1.5)

        if self.training and self.stochastic_train:
            cat = torch.distributions.Categorical(logits=mix_logits)
            mix_idx = cat.sample()
        elif (not self.training) and self.stochastic_eval:
            cat = torch.distributions.Categorical(logits=mix_logits)
            mix_idx = cat.sample()
        else:
            mix_idx = mix_logits.argmax(dim=-1)
        gather_idx = mix_idx.unsqueeze(1)
        mu = mu_all.gather(1, gather_idx)
        log_std = log_std_all.gather(1, gather_idx)
        std = torch.exp(log_std)

        if (self.training and self.stochastic_train) or ((not self.training) and self.stochastic_eval):
            eps = torch.randn_like(std)
        else:
            eps = torch.zeros_like(std)
        z = mu + std * eps
        u = torch.sigmoid(z)
        omega = self.omega_min + (self.omega_max - self.omega_min) * u

        z_expand = z.expand_as(mu_all)
        log_pz_comp = self._gaussian_log_prob(z_expand, mu_all, log_std_all) + mix_log_probs
        log_pz = torch.logsumexp(log_pz_comp, dim=-1, keepdim=True)
        log_abs_det = (
            math.log(self.omega_max - self.omega_min)
            + torch.log(u.clamp(min=self._eps))
            + torch.log((1.0 - u).clamp(min=self._eps))
        )
        log_prob_omega = log_pz - log_abs_det

        entropy = self._gaussian_entropy(log_std)

        risk_logit = self.risk_head(h)
        value = self.value_head(h)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([float(timestep)], device=omega.device)
        timestep = timestep.to(device=omega.device, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        if timestep.numel() == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        # normalized by common DDPM training horizon (1000)
        t_norm = (timestep / 999.0).unsqueeze(1).clamp(0.0, 1.0)
        prior_mu = self._prior_mean_z(t_norm)
        prior_log_std = torch.full_like(prior_mu, math.log(self.prior_std))

        self._step_infos.append(
            {
                "timestep_index": torch.tensor(
                    float(timestep_index if timestep_index is not None else -1),
                    device=omega.device,
                ),
                "omega": omega,
                "log_prob": log_prob_omega,
                "entropy": entropy,
                "risk_logit": risk_logit,
                "value": value,
                "mu": mu,
                "log_std": log_std,
                "prior_mu": prior_mu,
                "prior_log_std": prior_log_std,
            }
        )

        return omega
