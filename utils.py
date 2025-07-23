import torch


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
    feats = feats.to(torch.float64)
    mu = feats.mean(dim=0)
    Xc = feats - mu

    Sigma = (Xc.T @ Xc) / (N - 1)

    eigvals, _ = torch.linalg.eigh(Sigma)
    eigvals = eigvals.clamp_min(1e-12).to(torch.float32)
    return eigvals
