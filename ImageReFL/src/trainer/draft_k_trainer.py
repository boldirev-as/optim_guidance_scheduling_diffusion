import torch

from src.constants.dataset import DatasetColumns
from src.trainer.base_trainer import BaseTrainer


class DraftKTrainer(BaseTrainer):
    """
    Trainer class.
    Reproduce DraftK training.
    """

    def _get_dense_anchor_steps(self) -> list[int]:
        anchor_steps = self.cfg_trainer.get("dense_reward_anchor_steps", [])
        if not anchor_steps:
            return []
        grad_end = int(self.cfg_trainer.first_steps_count + self.cfg_trainer.k_steps)
        deduped = sorted(
            {
                int(anchor_step)
                for anchor_step in anchor_steps
                if self.cfg_trainer.first_steps_count < int(anchor_step) <= grad_end
            }
        )
        return deduped

    def _get_dense_anchor_weights(self, anchor_steps: list[int]) -> list[float]:
        raw_weights = self.cfg_trainer.get("dense_reward_weights", None)
        if raw_weights is None:
            return [1.0 / len(anchor_steps)] * len(anchor_steps)
        if len(raw_weights) != len(anchor_steps):
            raise ValueError("dense_reward_weights must have the same length as dense_reward_anchor_steps")
        total = float(sum(float(weight) for weight in raw_weights))
        if total <= 0:
            raise ValueError("dense_reward_weights must sum to a positive value")
        return [float(weight) / total for weight in raw_weights]

    def _has_dense_reward(self) -> bool:
        return len(self._get_dense_anchor_steps()) > 0

    def _sample_image_train(self, batch: dict[str, torch.Tensor]):
        total_steps = self.cfg_trainer.first_steps_count + self.cfg_trainer.k_steps
        self.model.set_timesteps(
            total_steps,
            device=self.device,
        )
        batch["guidance_min_step"] = self.cfg_trainer.first_steps_count
        batch["guidance_max_step"] = total_steps

        no_grad_steps = self.cfg_trainer.first_steps_count
        grad_steps = total_steps

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

        if not self._has_dense_reward():
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
            return

        anchor_steps = self._get_dense_anchor_steps()
        anchor_images: dict[int, torch.Tensor] = {}
        current_latents = latents

        for timestep_index in range(no_grad_steps, grad_steps):
            noise_pred = self.model.get_noise_prediction(
                latents=current_latents,
                timestep_index=timestep_index,
                encoder_hidden_states=encoder_hidden_states,
                do_classifier_free_guidance=self.cfg_trainer.do_classifier_free_guidance,
                detach_main_path=self.cfg_trainer.detach_main_path,
                batch=batch,
            )
            current_latents, pred_original_sample = self.model.sample_next_latents_and_pred_original(
                latents=current_latents,
                timestep_index=timestep_index,
                noise_pred=noise_pred,
            )
            anchor_step = timestep_index + 1
            if anchor_step in anchor_steps:
                pred_original_sample = pred_original_sample / self.model.vae.config.scaling_factor
                raw_image = self.model.vae.decode(pred_original_sample).sample
                reward_image = self.model.get_reward_image(raw_image)
                anchor_images[anchor_step] = reward_image
                if anchor_step == anchor_steps[-1]:
                    batch["image"] = reward_image

        batch["dense_reward_images"] = anchor_images

    def _score_train_batch(self, batch: dict[str, torch.Tensor]) -> None:
        if not self._has_dense_reward():
            super()._score_train_batch(batch=batch)
            return

        anchor_images = batch.pop("dense_reward_images", None)
        if not anchor_images:
            super()._score_train_batch(batch=batch)
            return

        anchor_steps = self._get_dense_anchor_steps()
        anchor_weights = self._get_dense_anchor_weights(anchor_steps)

        total_reward = 0.0
        final_anchor_step = anchor_steps[-1]
        final_reward = None
        final_clipped_reward = None

        for anchor_step, anchor_weight in zip(anchor_steps, anchor_weights):
            image = anchor_images[anchor_step].float()
            reward = self.train_reward_model._get_reward(batch, image)
            clipped_reward = self.train_reward_model._clip_reward(reward)

            batch["loss"] += (
                -(reward + self.train_reward_model.reward_offset)
                * self.train_reward_model.reward_scale_factor
                * anchor_weight
            ).mean()

            reward_mean = reward.mean().detach()
            batch[f"{self.train_reward_model.model_suffix}_anchor_{anchor_step}"] = reward_mean
            total_reward = total_reward + anchor_weight * reward_mean.item()

            if anchor_step == final_anchor_step:
                final_reward = reward_mean
                final_clipped_reward = clipped_reward.detach()

        if final_reward is not None:
            batch[self.train_reward_model.model_suffix] = final_reward
        batch[f"{self.train_reward_model.model_suffix}_dense_total"] = torch.tensor(
            total_reward, device=self.device
        )
        if final_clipped_reward is not None:
            batch[f"{self.train_reward_model.model_suffix}_clipped_mean"] = final_clipped_reward.mean()
            batch[f"{self.train_reward_model.model_suffix}_clipped_min"] = final_clipped_reward.min()
            batch[f"{self.train_reward_model.model_suffix}_clipped_max"] = final_clipped_reward.max()

    def _sample_image_eval(self, batch: dict[str, torch.Tensor]):
        total_steps = self.cfg_trainer.first_steps_count + self.cfg_trainer.k_steps
        self.model.set_timesteps(
            total_steps,
            device=self.device,
        )
        batch["guidance_min_step"] = self.cfg_trainer.first_steps_count
        batch["guidance_max_step"] = total_steps

        no_grad_steps = self.cfg_trainer.first_steps_count
        grad_steps = total_steps

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
