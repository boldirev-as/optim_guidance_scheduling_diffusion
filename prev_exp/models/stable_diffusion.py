import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, \
    DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTokenizer, CLIPTextModel

from prev_exp.models.guidance_schedulers import get_guidance_scheduler


class StableDiffusionWrapper:
    def __init__(self, model_version="runwayml/stable-diffusion-v1-5", device=None, num_steps=50):
        self.device = device
        self.dtype = torch.float16
        self.num_steps = num_steps

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_version, subfolder="scheduler", use_karras_sigmas=True)

        vae = AutoencoderKL.from_pretrained(
            model_version,
            subfolder="vae", torch_dtype=torch.float16
        ).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_version,
            subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_version, subfolder="text_encoder", torch_dtype=torch.float16
        ).to(self.device)

        unet = UNet2DConditionModel.from_pretrained(
            model_version, subfolder="unet", torch_dtype=torch.float16
        ).to(self.device)

        pipeline_kwargs = {
            'pretrained_model_name_or_path': model_version,
            'scheduler': scheduler,
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'unet': unet,
            'vae': vae,
            'torch_dtype': torch.float16,
            'requires_safety_checker': False
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(**pipeline_kwargs)
        self.pipe.unet.set_attn_processor(AttnProcessor2_0())

        self.pipe.scheduler.set_timesteps(num_steps, device=self.device)

        self.pipe.to(self.device)

        if self.device == 'cuda':
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
            self.pipe.enable_xformers_memory_efficient_attention()

        torch.backends.cuda.matmul.allow_tf32 = True

    @torch.inference_mode()
    def generate_images(
            self,
            prompts,
            num_images_per_prompt=1,
            guidance_scheduler_name="baseline",
            base_guidance=7.5,
            generator=None,
            height=512,
            width=512
    ):

        batch_size = len(prompts)

        guidance_scheduler_fn = get_guidance_scheduler(guidance_scheduler_name)
        self.pipe.scheduler.set_timesteps(self.num_steps, device=self.device)

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompts,
            negative_prompt=[''] * len(prompts),
            device=self.device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            prompt_embeds=None,
            negative_prompt_embeds=None
        )

        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(device=self.device, dtype=self.dtype)
        text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])

        latents_shape = (
            batch_size * num_images_per_prompt,
            self.pipe.unet.config.in_channels,
            height // self.pipe.vae_scale_factor,
            width // self.pipe.vae_scale_factor,
        )
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        latents = latents * self.pipe.scheduler.init_noise_sigma

        timesteps = self.pipe.scheduler.timesteps
        total_steps = len(timesteps)

        for i, t in enumerate(timesteps):
            current_guidance = guidance_scheduler_fn(i, total_steps, base_guidance)

            latent_model_input = latents.repeat(2, 1, 1, 1)
            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings
                ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + current_guidance * (noise_pred_text - noise_pred_uncond)

            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        latents = latents.to(self.pipe.vae.dtype)

        with torch.no_grad():
            image_pixels = self.pipe.vae.decode(latents).sample

        return image_pixels
