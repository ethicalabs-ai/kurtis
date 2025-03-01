# kurtis/model.py
import os
import click
import torch
from typing import Optional, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

# Determine torch data type based on CUDA capabilities.
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)

# Configure 4-bit quantization settings.
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch_dtype,
)


def load_model_and_tokenizer(
    config: object,
    model_name: Optional[str] = None,
    model_output: Optional[str] = None,
    checkpoint_name: str = "final_merged_checkpoint",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load the model and tokenizer automatically, or use the model_name if provided.

    Args:
        config: Configuration object with at least a TRANSFORMERS_MODEL_PRETRAINED attribute.
        model_name (str, optional): Name of the model to load. If not provided, defaults to config.TRANSFORMERS_MODEL_PRETRAINED.
        model_output (str, optional): Directory containing the model checkpoint. If it exists, the checkpoint subdirectory will be used.
        checkpoint_name (str): The name of the checkpoint directory. Defaults to "final_merged_checkpoint".

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer.
    """
    if not model_name:
        model_name = getattr(config, "TRANSFORMERS_MODEL_PRETRAINED", None)
        if model_name is None:
            raise ValueError("Config must have TRANSFORMERS_MODEL_PRETRAINED attribute")
    if model_output and os.path.exists(model_output):
        model_name = os.path.join(model_output, checkpoint_name)
    click.echo(f"Loading model and tokenizer: {model_name}")
    # Always load the tokenizer from the original pretrained model.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        getattr(config, "TRANSFORMERS_MODEL_PRETRAINED")
    )
    tokenizer.pad_token = tokenizer.eos_token
    chat_template = getattr(config, "CHAT_TEMPLATE", None)
    if chat_template:
        tokenizer.chat_template = chat_template
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=nf4_config,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
