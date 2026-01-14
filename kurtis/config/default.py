# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------

import os

from peft import LoraConfig, TaskType

# Model Configuration
# The base model to be fine-tuned
TRANSFORMERS_MODEL_PRETRAINED = "ibm-granite/granite-4.0-h-350m"

# Tokenizer used for preprocessing (data augmentation/cleaning)
PREPROCESSING_TOKENIZER_MODEL = TRANSFORMERS_MODEL_PRETRAINED

# Model name for inference
INFERENCE_MODEL = "ethicalabs/Kurtis-Granite-4.0-350m"

# Name of the fine-tuned model
MODEL_NAME = "Kurtis-Granite-4.0-350m-Instruct"
MODEL_DPO_NAME = "Kurtis-Granite-4.0-350m-Instruct-DPO"

# Hugging Face Hub Repository IDs
HF_REPO_ID = "ethicalabs/Kurtis-Granite-4.0-350m-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-Granite-4.0-350m-Instruct-DPO"

# Dataset Configuration
DATASET_NAME = "ethicalabs/kurtis-mental-health-v2-sft-reasoning"
DPO_DATASET_NAME = "mrs83/kurtis_mental_health_dpo"

# API Configuration
OPENAI_API_URL = "http://localhost:11434/v1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# Training Configuration (SFT)
TRAINING_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_split": "train",
    "prompt_column": "prompt",
    "response_column": "completion",
    "max_length": 1024,
    "num_train_epochs": 0.25,
    "warmup_ratio": 0.1,
    "batch_size": 8,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
    "logging_steps": 10,  # Frequent logging
    "eval_steps": 100,
    "eval_subset_size": 200,  # Validate on random 200 samples
}

# DPO Training Configuration
TRAINING_DPO_CONFIG = {
    "max_length": 1024,
    "num_train_epochs": 1,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}

# Evaluation Dataset Configuration
EVALUATION_DATASET = {
    "dataset_name": "ethicalabs/Kurtis-E1-SFT",
    "prompt_column": "question",
    "response_column": "answer",
    "dataset_max_samples": 500,
    "dataset_split": "validation",
    "max_length": 1024,
}

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=12,
    lora_alpha=24,
    lora_dropout=0.0075,
    target_modules=[
        "input_linear",
        "output_linear",
        "in_proj",
        "out_proj",
    ],
    bias="none",
    use_dora=True,
)

# Chat Template and Instructions
QA_INSTRUCTION = (
    "You are Kurtis, a compassionate and empathetic AI mental health support assistant developed by ethicalabs.ai. "
    "Your role is to provide thoughtful, supportive, and non-judgmental responses to help users explore their thoughts and feelings. "
    "\n\n"
    "Guidelines:\n"
    "- Listen actively and validate emotions without dismissing concerns\n"
    "- Ask clarifying questions to better understand the user's situation\n"
    "- Offer evidence-based coping strategies and perspectives when appropriate\n"
    "- Encourage professional help for serious concerns (crisis, self-harm, severe mental health issues)\n"
    "- Maintain appropriate boundaries - you are a supportive tool, not a licensed therapist\n"
    "- Be concise yet warm in your responses\n"
    "\n"
    "Remember: Your goal is to provide emotional support and helpful insights while encouraging users to seek professional care when needed."
)
