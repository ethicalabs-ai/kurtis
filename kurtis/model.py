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
from peft import AutoPeftModelForCausalLM, PeftConfig

# Determine torch data type based on CUDA capabilities.
torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)

# Configure 4-bit quantization settings.
bnb_config = (
    BitsAndBytesConfig(
        load_in_4bit=True,
    )
    if torch.cuda.is_available()
    else None
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
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer


def save_and_merge_model(
    final_checkpoint_dir: str,
    output_merged_dir: str,
    chat_template: str,
    hf_repo_id: str = "",
    push: bool = True,
):
    """Save the adapter model and tokenizer."""
    peft_config = PeftConfig.from_pretrained(final_checkpoint_dir)
    base_model_path = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if chat_template:
        tokenizer.chat_template = chat_template
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoint_dir, device_map="auto", torch_dtype=torch_dtype
    )
    model = model.merge_and_unload()
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    if push and hf_repo_id:
        model.push_to_hub(hf_repo_id, "Upload model")
        tokenizer.push_to_hub(hf_repo_id, "Upload tokenizer")
