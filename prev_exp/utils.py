import torch
import torch.nn.functional as F


def preprocess_for_inception(
        images: torch.Tensor,
        model_type: str = "torchvision",
) -> torch.Tensor:
    assert images.min() >= 0.0 and images.max() <= 1.0, "Ожидаем [0,1]!"

    x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

    if model_type == "torchvision":
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

    elif model_type == "pytorch_fid":
        pass
    else:
        raise ValueError("Unknown model_type")

    return x


def generate_images(unet, text_encoder, weight_dtype, device, noise_scheduler, vae, input_ids, generator=None):
    batch_size = input_ids.shape[0]

    encoder_hidden_states = text_encoder(input_ids.to(device))[0]
    latents = torch.randn((batch_size, 4, 64, 64), device=device,
                          dtype=weight_dtype, generator=generator)

    for t in noise_scheduler.timesteps:
        latent_in = noise_scheduler.scale_model_input(latents, t)
        noise_pred = unet(latent_in, t, encoder_hidden_states=encoder_hidden_states).sample
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    pred_original = latents.float() / vae.config.scaling_factor
    images = vae.decode(pred_original).sample
    images = (images / 2 + 0.5).clamp_(0, 1)

    return images


def calc_eigvals(features):
    feats = torch.stack(features, dim=0)
    N, D = feats.shape
    mu = feats.mean(dim=0)
    Xc = feats - mu
    Sigma = (Xc.T @ Xc) / (N - 1)

    eigvals, _ = torch.linalg.eigh(Sigma)  # ascending
    eigvals = eigvals.clamp_min(1e-12)
    eigvals = torch.sort(eigvals, descending=True).values  # enforce order
    return eigvals
