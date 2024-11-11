from peft import LoraConfig, TaskType

TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATA_AUGMENTATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
INFERENCE_MODEL = "mrs83/Kurtis-SmolLM2-1.7B-Instruct"
MODEL_NAME = "Kurtis-SmolLM2-1.7B-Instruct"
HF_REPO_ID = "mrs83/Kurtis-SmolLM2-1.7B-Instruct"

TRAINING_CONFIG = {
    "name": "mrs83/kurtis_mental_health_final",
    "split": "train",
    "max_samples": 10000,
    "max_length": 2048,
    "num_train_epochs": 2,
    "warmup_ratio": 0.3,
    "batch_size": 8,
    "lr": 3e-4,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}
DATASETS_CONFIG = {
    "kurtis_mental_health": {
        "name": "datasets/kurtis_mental_health",
        "text_column": "Context",
        "response_column": "Response",
        "split": "train",
        "max_samples": 4500,
        "max_length": 512,
    },
    "marmikpandya_mental_health": {
        "name": "marmikpandya/mental-health",
        "text_column": "input",
        "response_column": "output",
        "split": "train",
        "max_samples": 3500,
        "max_length": 512,
    },
    "fadodr_mental_health_therapy": {
        "name": "fadodr/mental_health_therapy",
        "text_column": "input",
        "response_column": "output",
        "split": "train",
        "max_samples": 3500,
        "max_length": 512,
    },
}

FINAL_DATASETS = {
    "datasets/kurtis_mental_health": "mrs83/kurtis_mental_health",
    "datasets/kurtis_mental_health_initial": "mrs83/kurtis_mental_health_initial",
    "datasets/kurtis_mental_health_final": "mrs83/kurtis_mental_health_final",
}

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
