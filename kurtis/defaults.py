from dataclasses import MISSING, dataclass, field, fields
from typing import List, Dict


@dataclass
class TrainingConfig:
    """
    # Usage Example
    cfg = config.TRAINING_CONFIG  # Assuming this is a dictionary
    training_config = TrainingConfig.from_dict(cfg)

    # Access parameters
    print(f"Warmup Ratio: {training_config.warmup_ratio}")
    print(f"Learning Rate: {training_config.lr}")
    """

    dataset_name: str = ""
    dataset_domain: str = ""
    dataset_subset: str = ""
    dataset_split: str = ""
    prompt_column: str = "prompt"
    response_column: str = "response"
    dataset_max_samples: int = 0
    bf16: bool = False
    tf32: bool = False
    checkpointing: bool = True
    warmup_ratio: float = 0.03
    num_train_epochs: int = 2
    batch_size: int = 1
    max_length: int = 2048
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    lr: float = 3e-4
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    weight_decay: float = 2e-2
    accumulation_steps: int = 8
    final_checkpoint_name: str = "final_checkpoint"
    final_merged_checkpoint_name: str = "final_merged_checkpoint"
    dpo_final_checkpoint_name: str = "dpo_final_checkpoint"
    dpo_final_merged_checkpoint_name: str = "dpo_final_merged_checkpoint"
    dataset_select: List[Dict] = field(default_factory=list)
    dataset_config_path: str = ""

    @classmethod
    def from_dict(cls, cfg: dict):
        init_kwargs = {}
        for field_obj in fields(cls):
            if field_obj.name in cfg:
                init_kwargs[field_obj.name] = cfg[field_obj.name]
            elif field_obj.default is not MISSING:
                init_kwargs[field_obj.name] = field_obj.default
            elif field_obj.default_factory is not MISSING:
                init_kwargs[field_obj.name] = field_obj.default_factory()
        return cls(**init_kwargs)
