import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights

from src.reward_models.base_model import BaseModel

from src.datasets.collate import collate_fn
from tqdm.auto import tqdm

MODEL_SUFFIX = "LSCD"


class LSCD(BaseModel):
    def __init__(self, images_per_batch: int, do_classifier_free_guidance: bool, device: torch.device):
        super().__init__(model_suffix=MODEL_SUFFIX, reward_scale_factor=0.1, reward_offset=0)
        self.images_per_batch = images_per_batch
        self.device = device
        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.ref_eigvals = None  # torch tensor on CPU
        self.cur_eigvals = None  # torch tensor on CPU

        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.inception.fc = torch.nn.Identity()
        self.inception.eval().requires_grad_(False).to(self.device)

    @torch.no_grad()
    def calc_eigenvalues(self, model, logger=None, eps: float = 1e-10):
        prompts = [p.strip() for p in open("prompts.txt") if p.strip()]
        seeds = list(range(15))

        feats_list = []

        for seed in tqdm(seeds):
            for start in range(0, len(prompts), self.images_per_batch):
                cur_prompts = prompts[start:start + self.images_per_batch]

                batch = []
                for prompt in cur_prompts:
                    batch.append(model.tokenize(prompt))
                    batch[-1]['tokenized_text'] = batch[-1]['tokenized_text'].to(self.device)

                batch = collate_fn(batch)

                imgs = model.sample_image(
                    latents=None,
                    start_timestep_index=0,
                    end_timestep_index=40,
                    batch=batch,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    seed=seed
                )

                imgs = (imgs + 1) / 2
                imgs = torch.clamp(imgs, 0, 1)

                # InceptionV3 expects 299x299; normalize w.r.t. its weights
                imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
                weights = Inception_V3_Weights.DEFAULT
                tf = weights.transforms()
                mean = torch.tensor(tf.mean, device=imgs.device).view(1, 3, 1, 1)
                std = torch.tensor(tf.std, device=imgs.device).view(1, 3, 1, 1)
                imgs = (imgs - mean) / std

                feats = self.inception(imgs)  # (B, D)
                feats_list.append(feats.detach().cpu())

        # (N, D) float64 for numerical stability
        emb = torch.cat(feats_list, dim=0).to(torch.float64)  # CPU tensor
        N, D = emb.shape

        # Center features to match E[(x-μ)(x-μ)^T]
        emb = emb - emb.mean(dim=0, keepdim=True)

        # Population covariance: (1/N) X^T X
        cov = (emb.T @ emb) / max(N, 1)

        # Ridge for stability
        cov += eps * torch.eye(D, dtype=cov.dtype, device=cov.device)

        # Symmetric eigendecomposition
        eigvals = torch.linalg.eigvalsh(cov)

        # Make strictly positive before log
        eigvals = torch.clamp(eigvals, min=eps)

        if self.ref_eigvals is None:
            self.ref_eigvals = eigvals.clone()  # CPU torch tensor
            self.cur_eigvals = eigvals.clone()
        else:
            self.cur_eigvals = eigvals

    @torch.no_grad()
    def _get_reward(self, model: BaseModel, logger=None, *args, **kwargs) -> float | int:
        self.calc_eigenvalues(model, logger)

        # Log-spectral distance with safe logs
        lscd_sum = torch.sum((torch.log(self.ref_eigvals) - torch.log(self.cur_eigvals)) ** 2).item()
        return lscd_sum
