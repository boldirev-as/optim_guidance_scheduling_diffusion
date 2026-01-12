import random
from contextlib import nullcontext

import torch

from src.constants.dataset import DatasetColumns
from src.trainer.base_trainer import BaseTrainer


class ReFLTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce ReFL training.
    """

    def _sample_image_train(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )
        batch["mid_timestep"] = mid_timestep

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

        base_scale = 7.5
        lambda_reg = 0.01
        # batch["loss"] += lambda_reg * ((self.model.last_omegas - base_scale) ** 2).mean()
        # print("loss guidance", batch["loss"])

    def _sample_image_eval(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(self.cfg_trainer.max_mid_timestep, device=self.device)

        mid_timestep = (
            random.randint(
                self.cfg_trainer.min_mid_timestep,
                self.cfg_trainer.max_mid_timestep - 1,
            )
            if self.is_train
            else self.cfg_trainer.max_mid_timestep - 1
        )
        batch["mid_timestep"] = mid_timestep

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
