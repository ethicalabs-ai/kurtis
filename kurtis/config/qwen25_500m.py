from .default import *  # noqa: F403
from peft import LoraConfig, TaskType


TRANSFORMERS_MODEL_PRETRAINED = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "ethicalabs/Kurtis-E1.1-Qwen2.5-0.5B-Instruct"
MODEL_NAME = "Kurtis-E1.1-Qwen2.5-0.5B-Instruct"
MODEL_DPO_NAME = "Kurtis-E1.1-Qwen2.5-0.5B-Instruct-DPO"
HF_REPO_ID = "ethicalabs/Kurtis-E1.1-Qwen2.5-0.5B-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-E1.1-Qwen2.5-0.5B-Instruct-DPO"
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.075,
    target_modules=[
        "down_proj",
        "gate_proj",
        "up_proj",
        # "k_proj",
        "o_proj",
        # "q_proj",
        # "v_proj",
    ],
    bias="none",
    use_dora=True,
)
TRAINING_CONFIG = {
    "dataset_name": "ethicalabs/Kurtis-E1-SFT",
    "dataset_split": "train",
    "dataset_max_samples": 10000,
    "prompt_column": "question",
    "response_column": "answer",
    "max_length": 1024,
    "num_train_epochs": 2,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
TRAINING_DPO_CONFIG = {
    "dataset_max_samples": 2000,
    "max_length": 1024,
    "num_train_epochs": 1,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
