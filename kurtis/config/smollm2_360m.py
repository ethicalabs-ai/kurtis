from .default import *  # noqa: F403
from peft import LoraConfig, TaskType


TRANSFORMERS_MODEL_PRETRAINED = "HuggingFaceTB/SmolLM2-360M-Instruct"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "ethicalabs/Kurtis-SmolLM2-360M-Instruct"
MODEL_NAME = "Kurtis-SmolLM2-360M-Instruct"
MODEL_DPO_NAME = "Kurtis-SmolLM2-360M-Instruct-DPO"
HF_REPO_ID = "ethicalabs/Kurtis-SmolLM2-360M-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-SmolLM2-360M-Instruct-DPO"
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
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
