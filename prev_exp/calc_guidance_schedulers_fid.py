import sys
from math import ceil
import platform

import torch
from tqdm import tqdm

from data.coco_loader import load_coco_data_batched
from models.stable_diffusion import StableDiffusionWrapper
from prev_exp.evaluation.metrics import Evaluator

if torch.cuda.is_available():
    device = 'cuda'
elif sys.platform == "darwin" or platform.system() == "Darwin":
    device = 'mps'
else:
    device = 'cpu'

num_gpus = torch.cuda.device_count() if device == "cuda" else 1
BATCH_SIZE = 4 if device != "cuda" else 32
NUM_SAMPLES = 4 if device != "cuda" else 10000
TOTAL_STEPS = ceil(NUM_SAMPLES / BATCH_SIZE)
IMG_SIZE = 512
NUM_STEPS = 25


def main():
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")

    print("Loading COCO data in batches...")
    coco_data = load_coco_data_batched(
        split='val',
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    print("Initializing models...")
    sd = StableDiffusionWrapper(device=device, num_steps=NUM_STEPS)
    evaluator = Evaluator(device=device)

    # 1 3 5 7 9 11 13 15 20
    guidance_types = [
                         ("baseline", 1), ("linear", 1), ("cosine", 1),
                         ("baseline", 3), ("linear", 3), ("cosine", 3),
                         ("baseline", 5), ("linear", 5), ("cosine", 5),
                         ("baseline", 7), ("linear", 7), ("cosine", 7),
                         ("baseline", 9), ("linear", 9), ("cosine", 9),
                         ("baseline", 11), ("linear", 11), ("cosine", 11),
                         ("baseline", 13), ("linear", 13), ("cosine", 13),
                         ("baseline", 15), ("linear", 15), ("cosine", 15),
                         ("baseline", 20), ("linear", 20), ("cosine", 20)][::-1]

    # guidance_types = [("baseline", 20), ("linear", 20), ("cosine", 20)]

    results = []

    for real_images, _ in tqdm(coco_data['batches'], total=TOTAL_STEPS):
        evaluator.fid.update(
            (real_images * 255).clamp(0, 255).to(torch.uint8).to('cuda' if device == 'cuda' else 'cpu'),
            real=True
        )

    generator = torch.Generator(device=device)

    for guidance, guidance_strength in guidance_types:
        print(f"\nEvaluating {guidance} guidance scheduler...")

        evaluator.fid.reset()

        clip_scores = []
        batch_count = 0

        coco_data = load_coco_data_batched(
            split='val',
            num_samples=NUM_SAMPLES,
            batch_size=BATCH_SIZE,
            image_size=IMG_SIZE
        )

        for _, captions in tqdm(coco_data['batches'], total=TOTAL_STEPS):
            batch_count += 1

            generated_images = sd.generate_images(
                prompts=captions,
                guidance_scheduler_name=guidance,
                num_images_per_prompt=1,
                generator=generator,
                height=IMG_SIZE,
                width=IMG_SIZE,
                base_guidance=guidance_strength
            )

            # torchvision.utils.save_image((generated_images[0] / 2 + 0.5).clamp(0, 1),
            #                              f"{guidance}_gen_images_{batch_count}.png")

            clip_images = (generated_images / 2 + 0.5).clamp(0, 1)

            clip_scores.extend(evaluator.compute_clip_score_batch(clip_images, captions))

            evaluator.fid.update(
                (clip_images * 255).clamp(0, 255).to(torch.uint8).to(
                    'cuda' if device == 'cuda' else 'cpu'),
                real=False
            )

        fid_score = evaluator.fid.compute().item()
        avg_clip_score = sum(clip_scores) / len(clip_scores)

        # diversity_images = []
        # diversity_prompts = coco_data['long_captions'] + coco_data['short_captions']
        #
        # diversity_batch_size = max(BATCH_SIZE // 5, 1)
        # for i in tqdm(range(0, len(diversity_prompts), diversity_batch_size), desc="Diversity Images"):
        #     prompt_batch = diversity_prompts[i:i + diversity_batch_size]
        #     diversity_images.extend(sd.generate_images(
        #         prompt_batch,
        #         guidance_scheduler_name=guidance,
        #         num_images_per_prompt=5,
        #         generator=generator,
        #         height=IMG_SIZE,
        #         width=IMG_SIZE
        #     ))
        #
        # diversity_score = evaluator.compute_diversity(diversity_images)

        print(guidance, guidance_strength, fid_score, type(fid_score), avg_clip_score,
              type(avg_clip_score))

        results.append({
            'guidance': guidance,
            'fid': fid_score,
            'clip_score': avg_clip_score
        })

    # plot_results(results)
    print("\nEvaluation complete!")
    print("Final Results:")
    print(results)
    for result in results:
        print(
            f"{result['guidance']:8s} | FID: {result['fid']:6.2f} | CLIP-Score: {result['clip_score']:.3f}")


if __name__ == "__main__":
    main()
