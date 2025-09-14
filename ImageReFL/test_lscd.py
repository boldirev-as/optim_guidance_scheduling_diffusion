import warnings
import os

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from PIL import Image  # NEW

from src.reward_models.lscd import LSCD
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(
    version_base=None, config_path="src/configs", config_name="combined_inference"
)
def main(config):
    device = torch.device('cuda')
    set_random_seed(0)

    model = instantiate(config.model).to(device)

    metric = LSCD(5, True, torch.device('cuda'))
    metric._get_reward(model)

    for num in [5, 10, 15, 20]:
        checkpoint = torch.load(f'saved/test_guidance_saved/checkpoint-epoch{num}.pth', device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        print('Epoch', num, metric._get_reward(model))


if __name__ == "__main__":
    main()
