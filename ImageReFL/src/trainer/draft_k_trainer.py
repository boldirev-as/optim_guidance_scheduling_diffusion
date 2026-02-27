import torch

from src.constants.dataset import DatasetColumns
from src.trainer.base_trainer import BaseTrainer


class DraftKTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce DraftK training.
    """

    def _sample_image_train(self, batch: dict[str, torch.Tensor]):
        self.model.set_timesteps(
            self.cfg_trainer.first_steps_count + self.cfg_trainer.k_steps,
            device=self.device,
        )
        batch["guidance_min_step"] = self.cfg_trainer.first_steps_count
        batch["guidance_max_step"] = self.cfg_trainer.first_steps_count + self.cfg_trainer.k_steps

        no_grad_steps = self.cfg_trainer.first_steps_count
        grad_steps = no_grad_steps + self.cfg_trainer.k_steps

        with torch.no_grad():
            latents, encoder_hidden_states = self.model.do_k_diffusion_steps(
                latents=None,
                start_timestep_index=0,
                batch=batch,
                end_timestep_index=no_grad_steps,
                return_pred_original=False,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                detach_main_path=self.cfg_trainer.detach_main_path,
                seed=batch.get("seeds", [None])[0],
            )

        batch["image"] = self.model.sample_image(
            latents=latents,
            start_timestep_index=no_grad_steps,
            end_timestep_index=grad_steps,
            encoder_hidden_states=encoder_hidden_states,
            batch=batch,
            do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
            detach_main_path=self.cfg_trainer.detach_main_path,
            seed=batch.get("seeds", [None])[0],
        )

    def _sample_image_eval(self, batch: dict[str, torch.Tensor]):
        self._sample_image_train(batch)
