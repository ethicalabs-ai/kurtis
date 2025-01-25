from .default import *  # noqa: F403
from peft import LoraConfig, TaskType


TRANSFORMERS_MODEL_PRETRAINED = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "mrs83/Kurtis-Qwen2.5-0.5B-Instruct"
MODEL_NAME = "Kurtis-Qwen2.5-0.5B-Instruct"
MODEL_DPO_NAME = "Kurtis-Qwen2.5-0.5B-Instruct-DPO"
HF_REPO_ID = "mrs83/Kurtis-Qwen2.5-0.5B-Instruct"
HF_DPO_REPO_ID = "mrs83/Kurtis-Qwen2.5-0.5B-Instruct-DPO"
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=12,
    lora_alpha=20,
    lora_dropout=0.05,
    target_modules=[
        "down_proj",
        "gate_proj",
        "up_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "v_proj",
    ],
    bias="none",
    use_dora=True,
)
