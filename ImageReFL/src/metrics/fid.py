import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance

from src.constants.dataset import DatasetColumns


class FIDMetric(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        model_suffix: str = "FID",
        feature: int = 2048,
    ):
        super().__init__()
        self.model_suffix = model_suffix
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.metric = FrechetInceptionDistance(feature=feature).to(device)
        self._seen_real = 0
        self._seen_fake = 0

    def _to_uint8(self, x: torch.Tensor) -> torch.Tensor:
        # Torchmetrics FID expects uint8 tensors in [0, 255], shape [B, C, H, W].
        if x.dtype == torch.uint8:
            return x.to(self.device, non_blocking=True)

        x = x.float()
        # Support both normalized [-1, 1] and image-space [0, 1] ranges.
        if x.min().item() < 0.0:
            x = x * 0.5 + 0.5
        x = x.clamp(0, 1)
        x = (x * 255.0).round().to(torch.uint8)
        return x.to(self.device, non_blocking=True)

    def reset(self) -> None:
        self.metric.reset()
        self._seen_real = 0
        self._seen_fake = 0

    def update(self, batch: dict, model=None) -> None:
        if DatasetColumns.original_image.name not in batch:
            return
        if "raw_image" in batch:
            fake = batch["raw_image"]
        elif "image" in batch:
            fake = batch["image"]
        else:
            return
        real = batch[DatasetColumns.original_image.name]

        fake = self._to_uint8(fake)
        real = self._to_uint8(real)

        self.metric.update(real, real=True)
        self.metric.update(fake, real=False)
        self._seen_real += real.shape[0]
        self._seen_fake += fake.shape[0]

    @torch.no_grad()
    def _get_reward(self, model=None, logger=None, *args, **kwargs) -> float:
        if self._seen_real == 0 or self._seen_fake == 0:
            if logger is not None:
                logger.warning("FIDMetric: missing real/fake samples, returning NaN.")
            return float("nan")
        return float(self.metric.compute().item())
