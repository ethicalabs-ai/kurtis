from .default import *  # noqa: F403
from peft import LoraConfig, TaskType


TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/SmolLM2-135M-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
INFERENCE_MODEL = "ethicalabs/Kurtis-E1.1-SmolLM2-135M-Instruct"
MODEL_NAME = "Kurtis-E1.1-SmolLM2-135M-Instruct"
MODEL_DPO_NAME = "Kurtis-E1.1-SmolLM2-135M-Instruct-DPO"
HF_REPO_ID = "ethicalabs/Kurtis-E1.1-SmolLM2-135M-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-E1.1-SmolLM2-135M-Instruct-DPO"
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=[
        "down_proj",
        # "gate_proj",
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
    "prompt_column": "question",
    "response_column": "answer",
    "dataset_max_samples": 10000,
    "max_length": 512,
    "num_train_epochs": 2,
    "warmup_ratio": 0.2,
    "batch_size": 4,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
TRAINING_DPO_CONFIG = {
    "dataset_max_samples": 2500,
    "max_length": 1024,
    "num_train_epochs": 1,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
