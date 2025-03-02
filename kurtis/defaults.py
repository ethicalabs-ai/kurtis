from dataclasses import dataclass, fields


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
    dataset_subset: str = ""
    dataset_split: str = ""
    prompt_column: str = "prompt"
    response_column: str = "response"
    dataset_max_samples: int = 0
    bf16: bool = True
    tf32: bool = True
    checkpointing: bool = True
    warmup_ratio: float = 0.03
    num_train_epochs: int = 2
    batch_size: int = 1
    max_length: int = 2048
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    lr: float = 3e-4
    lr_scheduler_type: str = "cosine"
    optim: str = "paged_adamw_8bit"
    weight_decay: float = 2e-2
    accumulation_steps: int = 8
    final_checkpoint_name: str = "final_checkpoint"
    final_merged_checkpoint_name: str = "final_merged_checkpoint"
    dpo_final_checkpoint_name: str = "dpo_final_checkpoint"
    dpo_final_merged_checkpoint_name: str = "dpo_final_merged_checkpoint"

    @classmethod
    def from_dict(cls, cfg: dict):
        init_kwargs = {
            field.name: cfg.get(field.name, getattr(cls, field.name))
            for field in fields(cls)
        }
        return cls(**init_kwargs)
