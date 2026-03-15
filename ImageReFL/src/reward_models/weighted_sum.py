import typing as tp

import torch
from hydra.utils import instantiate

from src.reward_models.base_model import BaseModel


class WeightedSumReward(BaseModel):
    def __init__(
            self,
            reward_models: list[BaseModel] | list[tp.Any],
            device: torch.device | str,
            weights: list[float] | None = None,
            model_suffix: str = "WeightedReward",
            reward_scale_factor: float = 1.0,
            reward_offset: float = 0.0,
            reward_clip_min: float | None = None,
            reward_clip_max: float | None = None,
            normalize_weights: bool = True,
    ):
        super().__init__(
            model_suffix=model_suffix,
            reward_scale_factor=reward_scale_factor,
            reward_offset=reward_offset,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )

        if len(reward_models) == 0:
            raise ValueError("reward_models must contain at least one reward model.")

        instantiated_models = []
        for reward_model in reward_models:
            if isinstance(reward_model, BaseModel):
                instantiated_models.append(reward_model)
            else:
                instantiated_models.append(instantiate(reward_model, device=device))

        self.reward_models = torch.nn.ModuleList(instantiated_models)
        self.component_model_suffixes = [
            reward_model.model_suffix for reward_model in self.reward_models
        ]

        if weights is None:
            weights = [1.0] * len(self.reward_models)
        if len(weights) != len(self.reward_models):
            raise ValueError("weights must have the same length as reward_models.")

        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if normalize_weights:
            weight_sum = float(weight_tensor.sum().item())
            if weight_sum == 0.0:
                raise ValueError("weights sum to zero; cannot normalize.")
            weight_tensor = weight_tensor / weight_sum
        self.register_buffer("weights", weight_tensor)

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        tokenized = {}
        for reward_model in self.reward_models:
            tokenized.update(reward_model.tokenize(caption))
        return tokenized

    def _compute_component_rewards(
            self,
            batch: tp.Dict[str, torch.Tensor],
            image: torch.Tensor,
            *args, **kwargs
    ) -> tuple[torch.Tensor, list[tuple[str, torch.Tensor]]]:
        component_rewards = []
        for reward_model in self.reward_models:
            reward = reward_model._get_reward(batch, image, *args, **kwargs)
            component_rewards.append((reward_model.model_suffix, reward))

        stacked_rewards = torch.stack(
            [reward.float() for _, reward in component_rewards], dim=0
        )
        weights = self.weights.to(device=stacked_rewards.device, dtype=stacked_rewards.dtype)
        combined_reward = (stacked_rewards * weights[:, None]).sum(dim=0)
        return combined_reward, component_rewards

    def _get_reward(
            self,
            batch: tp.Dict[str, torch.Tensor],
            image: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        combined_reward, _ = self._compute_component_rewards(batch, image, *args, **kwargs)
        return combined_reward

    def score_grad(
            self,
            batch: tp.Dict[str, torch.Tensor],
    ) -> None:
        image = batch["image"].float()
        reward, component_rewards = self._compute_component_rewards(batch, image)
        clipped_reward = self._clip_reward(reward)
        loss = -(reward + self.reward_offset) * self.reward_scale_factor
        batch["loss"] += loss.mean()
        batch[self.model_suffix] = reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_mean"] = clipped_reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_min"] = clipped_reward.min().detach()
        batch[f"{self.model_suffix}_clipped_max"] = clipped_reward.max().detach()

        for model_suffix, component_reward in component_rewards:
            batch[model_suffix] = component_reward.mean().detach()

    def score(
            self,
            batch: tp.Dict[str, torch.Tensor],
            *args, **kwargs
    ) -> None:
        image = batch["image"].float()
        with torch.no_grad():
            reward, component_rewards = self._compute_component_rewards(batch, image, *args, **kwargs)
        clipped_reward = self._clip_reward(reward)
        batch[self.model_suffix] = reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_mean"] = clipped_reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_min"] = clipped_reward.min().detach()
        batch[f"{self.model_suffix}_clipped_max"] = clipped_reward.max().detach()

        for model_suffix, component_reward in component_rewards:
            batch[model_suffix] = component_reward.mean().detach()
