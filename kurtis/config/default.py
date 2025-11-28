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
TRANSFORMERS_MODEL_PRETRAINED = "google/gemma-3-270m-it"

# Tokenizer used for preprocessing (data augmentation/cleaning)
PREPROCESSING_TOKENIZER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

# Model name for inference
INFERENCE_MODEL = "ethicalabs/Kurtis-Gemma-3-270m-it"

# Name of the fine-tuned model
MODEL_NAME = "Kurtis-Gemma-3-270m-it-Instruct"
MODEL_DPO_NAME = "Kurtis-Gemma-3-270m-it-Instruct-DPO"

# Hugging Face Hub Repository IDs
HF_REPO_ID = "ethicalabs/Kurtis-Gemma-3-270m-it-Instruct"
HF_DPO_REPO_ID = "ethicalabs/Kurtis-Gemma-3-270m-it-Instruct-DPO"

# Dataset Configuration
DATASET_NAME = "ethicalabs/kurtis-mental-health-v2-sft-reasoning"
DPO_DATASET_NAME = "mrs83/kurtis_mental_health_dpo"

# API Configuration
OPENAI_API_URL = "http://localhost:11434/v1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")

# Translation Configuration
TRANSLATION_GGUF_MODEL = "hf.co/mradermacher/TowerInstruct-7B-v0.2-GGUF:IQ4_XS"
TRANSLATION_LANGUAGES = [
    "Italian",
    "Spanish",
    "German",
    "French",
    "Portuguese",
    "Dutch",
]

# Training Configuration (SFT)
TRAINING_CONFIG = {
    "dataset_name": DATASET_NAME,
    "dataset_split": "train",
    "prompt_column": "question",
    "response_column": "answer",
    "max_length": 1024,
    "num_train_epochs": 3,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 5e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}

# DPO Training Configuration
TRAINING_DPO_CONFIG = {
    "max_length": 1024,
    "num_train_epochs": 2,
    "warmup_ratio": 0.2,
    "batch_size": 8,
    "lr": 1e-5,
    "accumulation_steps": 1,
    "weight_decay": 2e-2,
}

# Evaluation Dataset Configuration
EVALUATION_DATASET = {
    "dataset_name": DATASET_NAME,
    "prompt_column": "question",
    "response_column": "answer",
    "dataset_max_samples": 500,
    "split": "validation",
    "max_length": 1024,
}

# LoRA Configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=12,
    lora_alpha=24,
    lora_dropout=0.0075,
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

# Chat Template and Instructions
QA_INSTRUCTION = "You are a compassionate and empathetic mental-health assistant named Kurtis, trained by ethicalabs.ai.\nYou provide thoughtful and supportive responses to user queries."
CHAT_TEMPLATE = """
{{ bos_token }}
{%- if messages[0]['role'] == 'system' -%}
    {%- if messages[0]['content'] is string -%}
        {%- set first_user_prefix = messages[0]['content'] + '

' -%}
    {%- else -%}
        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '

' -%}
    {%- endif -%}
    {%- set loop_messages = messages[1:] -%}
{%- else -%}
    {%- set first_user_prefix = "" -%}
    {%- set loop_messages = messages -%}
{%- endif -%}
{%- for message in loop_messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{ raise_exception("Conversation roles must alternate user/assistant/user/assistant/...") }}
    {%- endif -%}
    {%- if (message['role'] == 'assistant') -%}
        {%- set role = "model" -%}
    {%- else -%}
        {%- set role = message['role'] -%}
    {%- endif -%}
    {{ '<start_of_turn>' + role + '
' + (first_user_prefix if loop.first else "") }}
    {%- if message['content'] is string -%}
        {{ message['content'] | trim }}
    {%- elif message['content'] is iterable -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'image' -%}
                {{ '<img>' }}
            {%- elif item['type'] == 'text' -%}
                {{ item['text'] | trim }}
            {%- endif -%}
        {%- endfor -%}
    {%- else -%}
        {{ raise_exception("Invalid content type") }}
    {%- endif -%}
    {{ '<end_of_turn>
' }}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{'<start_of_turn>model
'}}
{%- endif -%}
"""
