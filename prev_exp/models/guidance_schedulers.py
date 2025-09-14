import math


def baseline_scheduler(step: int, total_steps: int, overall_w: float):
    return overall_w


def get_t(step: int, total_steps: int):
    t = total_steps - 1 - step
    normalized_t = t / total_steps
    return t, normalized_t


def _shape_linear_increasing(normalized_t: float):
    return 1 - normalized_t


def linear_increasing_scheduler(step: int, total_steps: int, overall_w: float):
    alpha = step / (total_steps - 1)
    return overall_w * 2 * alpha


def _shape_cosine_increasing(normalized_t: float):
    return math.cos(math.pi * normalized_t) + 1.0


def cosine_increasing_scheduler(step: int, total_steps: int, overall_w: float):
    alpha = step / (total_steps - 1)
    return overall_w * (1 + math.cos(math.pi * (1 - alpha)))


def get_guidance_scheduler(name: str):
    schedulers = {
        "baseline": baseline_scheduler,
        "linear": linear_increasing_scheduler,
        "cosine": cosine_increasing_scheduler,
    }
    return schedulers[name]
