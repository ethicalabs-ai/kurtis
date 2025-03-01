from .default import *  # noqa: F403
from peft import LoraConfig, TaskType


TRANSFORMERS_MODEL_PRETRAINED = "PleIAs/Pleias-350m-Preview"
DATA_AUGMENTATION_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
INFERENCE_MODEL = "ethicalabs-ai/Kurtis-Pleias-350m-Instruct"
MODEL_NAME = "Kurtis-Pleias-350m-Instruct"
MODEL_DPO_NAME = "Kurtis-Pleias-350m-Instruct-DPO"
HF_REPO_ID = "ethicalabs-ai/Kurtis-Pleias-350m-Instruct"
HF_DPO_REPO_ID = "ethicalabs-ai/Kurtis-Pleias-350m-Instruct-DPO"
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
CHAT_TEMPLATE = """{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are an helpful and empathetic mental-health AI assistant named Kurtis, trained by ethicalabs.ai<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""
