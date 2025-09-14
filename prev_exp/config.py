from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf, DictConfig


@dataclass
class TrainConfig:
    # --- core ---
    grad_scale: float = 1
    input_perturbation: float = 0.0
    revision: Optional[str] = None
    non_ema_revision: Optional[str] = None

    # --- data ---
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    image_column: str = "image"
    caption_column: str = "text"
    max_train_samples: Optional[int] = None
    validation_prompts: List[str] = field(default_factory=list)

    # --- paths & I/O ---
    output_dir: str = "checkpoint/refl"
    cache_dir: Optional[str] = None
    logging_dir: str = "logs"

    # --- reproducibility ---
    seed: Optional[int] = None

    # --- image preprocessing ---
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False

    # --- training ---
    train_batch_size: int = 1
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = 10000
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    base_lr = 3e-6
    learning_rate: float = base_lr  # * train_batch_size * gradient_accumulation_steps
    scale_lr: bool = False
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 500
    snr_gamma: Optional[float] = None
    use_8bit_adam: bool = False
    ema_reward: float = 0.0

    # --- optimiser ---
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 0.5

    # --- misc/hub ---
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None
    report_to: str = "wandb"
    wandb_project: str = "text2image-refl"
    wandb_run_name: str = "refl"

    # --- distributed ---
    local_rank: int = -1

    # --- checkpointing ---
    checkpointing_steps: int = 1000
    evaluation_steps: int = 500
    eval_batch_size: int = 6
    validation_json: str = "pick-a-pic-v2-unique-prompts/test.json"
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None

    # --- memory/attention ---
    enable_xformers_memory_efficient_attention: bool = True
    noise_offset: float = 0.0

    # --- validation/logging ---
    validation_epochs: int = 5
    tracker_project_name: str = "text2image-refl"

    # --- dataloader ---
    dataloader_num_workers: int = 8


def get_cfg() -> DictConfig:
    """Merge defaults, YAML file (if any) and CLI overrides."""
    cfg = OmegaConf.structured(TrainConfig)

    cli = OmegaConf.from_cli()
    if "config" in cli:
        yaml_cfg = OmegaConf.load(cli.pop("config"))
        cfg = OmegaConf.merge(cfg, yaml_cfg)

    cfg = OmegaConf.merge(cfg, cli)

    cfg.non_ema_revision = cfg.revision

    return cfg
