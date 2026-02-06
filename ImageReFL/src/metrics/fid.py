import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from src.constants.dataset import DatasetColumns


class FIDMetric:
    def __init__(
        self,
        device: str = "cuda",
        model_suffix: str = "FID",
        feature: int = 2048,
    ):
        self.model_suffix = model_suffix
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = FrechetInceptionDistance(feature=feature).to(device)

    def reset(self) -> None:
        self.metric.reset()

    def update(self, batch: dict, model=None) -> None:
        if "raw_image" in batch:
            fake = batch["raw_image"]
        else:
            fake = batch["image"]
        real = batch[DatasetColumns.original_image.name]

        fake = fake.float()
        real = real.float()
        fake = (fake * 0.5 + 0.5).clamp(0, 1)
        real = (real * 0.5 + 0.5).clamp(0, 1)

        self.metric.update(real, real=True)
        self.metric.update(fake, real=False)

    @torch.no_grad()
    def _get_reward(self, model=None, logger=None, *args, **kwargs) -> float:
        return float(self.metric.compute().item())
