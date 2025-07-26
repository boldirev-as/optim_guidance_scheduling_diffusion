import torch
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from transformers.image_transforms import to_pil_image

from models.stable_diffusion import StableDiffusionWrapper

pretrained_model_name_or_path = 'CompVis/stable-diffusion-v1-4'
model = StableDiffusionWrapper(
    device='mps', model_version=pretrained_model_name_or_path, num_steps=25
)
#
# noise_scheduler = DDPMScheduler.from_pretrained(
#     pretrained_model_name_or_path, subfolder="scheduler"
# )
# tokenizer = CLIPTokenizer.from_pretrained(
#     pretrained_model_name_or_path, subfolder="tokenizer"
# )
# text_encoder = CLIPTextModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="text_encoder"
# )
# vae = AutoencoderKL.from_pretrained(
#     pretrained_model_name_or_path, subfolder="vae"
# )
# unet = UNet2DConditionModel.from_pretrained(
#     pretrained_model_name_or_path, subfolder="unet"
# )

# sd = torch.load('unet_step_600.pt', map_location='cpu')
# new_sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
#
# unet.load_state_dict(new_sd)
# model.pipe.unet = unet
# model.pipe.noise_scheduler = noise_scheduler
# model.pipe.tokenizer = tokenizer
# model.pipe.text_encoder = text_encoder
# model.pipe.vae = vae

prompt = ["a photo of an astronaut riding a horse"]
num_images_per_prompt = 1
guidance_scheduler_name = "baseline"
base_guidance = 7.5
generator = torch.Generator(device='mps').manual_seed(0)
height = 512
width = 512

images = model.generate_images(
    prompt,
    num_images_per_prompt=num_images_per_prompt,
    guidance_scheduler_name=guidance_scheduler_name,
    base_guidance=base_guidance,
    generator=generator,
    height=height,
    width=width
)

image = (images[0] / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu()
img = to_pil_image(image)
img.save("my_image.png")
