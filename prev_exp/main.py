import sys
from math import ceil
import platform
from pathlib import Path

# Ensure project root is on sys.path so `data` and other modules resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from data.coco_loader import load_coco_data_batched
from prev_exp.models.stable_diffusion import StableDiffusionWrapper
from prev_exp.evaluation.metrics import Evaluator

if torch.cuda.is_available():
    device = 'cuda'
elif sys.platform == "darwin" or platform.system() == "Darwin":
    device = 'mps'
else:
    device = 'cpu'

num_gpus = torch.cuda.device_count() if device == "cuda" else 1
BATCH_SIZE = 1
NUM_SAMPLES = 100  # только текстовые промпты, без загрузки датасетов
TOTAL_STEPS = ceil(NUM_SAMPLES / BATCH_SIZE)
IMG_SIZE = 512
NUM_STEPS = 10


def main():
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")

    print("Loading prompts...")
    prompts_file = PROJECT_ROOT / "prompts.txt"
    if prompts_file.exists():
        prompts = [p.strip() for p in prompts_file.read_text().splitlines() if p.strip()]
    else:
        prompts = []
    if not prompts:
        prompts = [
            "a photo of a cat on a skateboard",
            "a scenic mountain landscape at sunrise",
            "a futuristic city street at night with neon lights",
            "a portrait of an astronaut in a forest",
            "a bowl of ramen on a wooden table",
            "a cozy cabin in the snow",
            "a colorful parrot in a jungle",
            "a classic car parked near the beach",
        ]
    prompts = prompts[:NUM_SAMPLES]

    print("Initializing models...")
    sd = StableDiffusionWrapper(device=device, num_steps=NUM_STEPS)
    evaluator = Evaluator(device=device)

    # guidance combinations: baseline scales, cosine, linear
    scales = [1, 3, 5, 7, 9, 11, 13, 15, 20]
    guidance_types = (
        [("baseline", s) for s in scales]
        + [("cosine", s) for s in scales]
        + [("linear", s) for s in scales]
    )

    results = []

    generator = torch.Generator(device=device)

    baseline_ref_features = None
    all_images_for_diversity = []

    for guidance, guidance_strength in guidance_types:
        print(f"\nEvaluating {guidance} guidance scheduler...")

        evaluator.gen_features = []

        clip_scores = []
        hps_scores = []
        pick_scores = []
        batch_count = 0

        for start in tqdm(range(0, len(prompts), BATCH_SIZE), total=TOTAL_STEPS):
            captions = prompts[start:start + BATCH_SIZE]
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
            hps_scores.extend(evaluator.compute_hps_batch(clip_images, captions))
            pick_scores.extend(evaluator.compute_pickscore_batch(clip_images, captions))
            all_images_for_diversity.extend([img.detach().cpu() for img in clip_images])
            evaluator.add_generated_features(clip_images)

        avg_clip_score = sum(clip_scores) / len(clip_scores)
        avg_hps = sum(hps_scores) / len(hps_scores)
        avg_pick = sum(pick_scores) / len(pick_scores)
        avg_diversity = evaluator.compute_diversity(all_images_for_diversity)
        # Для LSCD используем фичи первого запуска как референс
        if baseline_ref_features is None:
            # clone list of tensors to avoid mutation
            baseline_ref_features = [t.clone() for t in evaluator.gen_features]
            evaluator.real_features = baseline_ref_features
        else:
            evaluator.real_features = baseline_ref_features
        lscd = evaluator.compute_lscd()

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
            'clip_score': avg_clip_score,
            'hps': avg_hps,
            'pickscore': avg_pick,
            'diversity': avg_diversity,
            'lscd': lscd,
        })

    # plot_results(results)
    print("\nEvaluation complete!")
    print("Final Results:")
    print(results)
    for result in results:
        print(
            f"{result['guidance']:8s} | CLIP-Score: {result['clip_score']:.3f} | "
            f"HPS: {result['hps']:.3f} | PickScore: {result['pickscore']:.3f} | "
            f"Diversity: {result['diversity']:.3f} | LSCD: {result['lscd']:.3f}"
        )


if __name__ == "__main__":
    main()
