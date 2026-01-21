import typing as tp
from abc import abstractmethod

import torch.utils.checkpoint


class BaseModel(torch.nn.Module):
    def __init__(
            self,
            model_suffix: str,
            reward_scale_factor: float,
            reward_offset: float,
            reward_clip_min: float | None = None,
            reward_clip_max: float | None = None,
    ):
        super().__init__()
        self.model_suffix = model_suffix
        self.reward_scale_factor = reward_scale_factor
        self.reward_offset = reward_offset
        self.reward_clip_min = reward_clip_min
        self.reward_clip_max = reward_clip_max

    def _clip_reward(self, reward: torch.Tensor) -> torch.Tensor:
        if self.reward_clip_min is None and self.reward_clip_max is None:
            return reward
        return reward.clamp(min=self.reward_clip_min, max=self.reward_clip_max)

    @abstractmethod
    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _get_reward(
            self,
            batch: tp.Dict[str, torch.Tensor],
            image: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        pass

    def score_grad(
            self,
            batch: tp.Dict[str, torch.Tensor],
    ) -> None:
        image = batch["image"].float()
        reward = self._get_reward(batch, image)
        clipped_reward = self._clip_reward(reward)
        loss = -(reward + self.reward_offset) * self.reward_scale_factor
        batch["loss"] += loss.mean()
        batch[self.model_suffix] = reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_mean"] = clipped_reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_min"] = clipped_reward.min().detach()
        batch[f"{self.model_suffix}_clipped_max"] = clipped_reward.max().detach()

    def score(
            self,
            batch: tp.Dict[str, torch.Tensor],
            *args, **kwargs
    ) -> None:
        image = batch["image"].float()
        with torch.no_grad():
            reward = self._get_reward(batch, image, *args, **kwargs)
        clipped_reward = self._clip_reward(reward)
        batch[self.model_suffix] = reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_mean"] = clipped_reward.mean().detach()
        batch[f"{self.model_suffix}_clipped_min"] = clipped_reward.min().detach()
        batch[f"{self.model_suffix}_clipped_max"] = clipped_reward.max().detach()
