from peft import LoraConfig, TaskType

TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "mrs83/Kurtis-SmolLM2-1.7B-Instruct"
MODEL_NAME = "Kurtis-SmolLM2-1.7B-Instruct"
HF_REPO_ID = "Kurtis-SmolLM2-1.7B-Instruct"

TRAINING_CONFIG = {
    "name": "mrs83/kurtis_mental_health_final",
    "split": "train",
    "max_samples": 10000,
    "max_length": 2048,
    "num_train_epochs": 3,
    "warmup_ratio": 0.2,
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


EVALUATION_DATASET = {
    "name": "Amod/mental_health_counseling_conversations",
    "text_column": "Context",
    "response_column": "Response",
    "max_samples": 500,
}


FINAL_DATASETS = {
    "datasets/kurtis_mental_health": "mrs83/kurtis_mental_health",
    "datasets/kurtis_mental_health_initial_v2": "mrs83/kurtis_mental_health_initial_v2",
    "datasets/kurtis_mental_health_final_v2": "mrs83/kurtis_mental_health_final_v2",
}
DPO_DATASETS = {
    "datasets/kurtis_mental_health_dpo/kurtis_clean.parquet": "mrs83/kurtis_mental_health_dpo"
}

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

QA_INSTRUCTION = "You are a compassionate and empathetic mental-health assistant, providing thoughtful and supportive responses to user queries."
