import typing as tp

import numpy as np
import torch
import torch.utils.checkpoint
from transformers import AutoImageProcessor, AutoModel

from src.reward_models.base_model import BaseModel

PROCESSOR_NAME = "facebook/dino-vits16"
MODEL_NAME = "facebook/dino-vits16"

MODEL_SUFFIX = "Diversity"


def compute_diversity(embeddings: torch.Tensor) -> torch.Tensor:
    """
    embeddings: Tensor of shape [N, D], assumed unit-normalized.
    Returns: scalar tensor with diversity score.
    """
    N = embeddings.size(0)
    if N < 2:
        return torch.tensor(0.0, device=embeddings.device)

    sum_vec = embeddings.sum(dim=0)
    sum_norm_sq = (sum_vec * sum_vec).sum()

    mean_pair_dot = (sum_norm_sq - N) / (N * (N - 1))

    diversity = 1.0 - mean_pair_dot
    return diversity


class Diversity(BaseModel):
    def __init__(self, images_per_batch: int, device: torch.device):
        super().__init__(
            model_suffix=MODEL_SUFFIX, reward_scale_factor=0.1, reward_offset=0
        )
        self.images_per_batch = images_per_batch
        self.device = device
        self.dino_model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    def tokenize(self, caption: str) -> tp.Dict[str, torch.Tensor]:
        return {}

    @torch.no_grad()
    def _get_reward(
            self,
            batch: tp.Dict[str, torch.Tensor],
            image: torch.Tensor,
            *args, **kwargs
    ) -> torch.Tensor:
        # device = image.device

        # processor = AutoImageProcessor.from_pretrained(PROCESSOR_NAME)

        all_feats = []
        for i in range(0, image.shape[0], self.images_per_batch):
            feats = self.dino_model(image[i: i + self.images_per_batch]).last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)

        embeddings = torch.cat(all_feats, dim=0)
        diversity = compute_diversity(embeddings)

        return diversity
