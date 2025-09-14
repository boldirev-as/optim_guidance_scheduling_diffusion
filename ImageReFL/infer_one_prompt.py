import warnings
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from PIL import Image  # NEW

from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="combined_inference"
)
def main(config):
    device = torch.device('cuda')
    set_random_seed(0)

    model = instantiate(config.model).to(device)
    model.do_guidance_w_loss = config.model.do_guidance_w_loss
    checkpoint = torch.load(config.inferencer.from_pretrained, device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    end_timestep_index = 40
    batch_size = 1
    do_classifier_free_guidance = True

    prompt = 'a photo of a cat'
    batch = model.tokenize(prompt)
    batch['tokenized_text'] = batch['tokenized_text'].to(device)

    model.set_timesteps(end_timestep_index, device=device)

    noised_latents = model.get_latents(batch_size=batch_size, device=device)

    encoder_hidden_states = model.get_encoder_hidden_states(
        batch=batch, do_classifier_free_guidance=do_classifier_free_guidance
    )

    # Collect selected frames here
    selected_frames = []

    for cur_timestamp in range(end_timestep_index):
        with torch.no_grad():
            pred_original_sample, _ = model.predict_next_latents(
                latents=noised_latents,
                timestep_index=cur_timestamp,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
                return_pred_original=True,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

            # Your original decoding step
            pred_original_sample = noised_latents / model.vae.config.scaling_factor
            raw_image = model.vae.decode(pred_original_sample).sample
            pil_image = model.get_pil_image(raw_image)[0]

            # Keep only frames where the step index is divisible by 5
            if cur_timestamp % 5 == 0:
                selected_frames.append(pil_image)

            noised_latents, _ = model.predict_next_latents(
                latents=noised_latents,
                timestep_index=cur_timestamp,
                encoder_hidden_states=encoder_hidden_states,
                batch=batch,
                return_pred_original=False,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

    # Make sure output directory exists
    os.makedirs('samples', exist_ok=True)

    # Stitch the selected frames into one long (horizontal) image
    if selected_frames:
        # Ensure consistent mode/size (model outputs should already match)
        widths, heights = zip(*(im.size for im in selected_frames))
        total_width = sum(widths)
        max_height = max(heights)

        frames_uniform = [im.convert('RGB') for im in selected_frames]

        long_image = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for im in frames_uniform:
            long_image.paste(im, (x_offset, 0))
            x_offset += im.width

        long_image.save('samples/cat_long_horizontal_guidance1.png')
    else:
        print("No frames selected; nothing to save.")


if __name__ == "__main__":
    main()
