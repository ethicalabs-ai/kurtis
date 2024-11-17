import os

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

torch_dtype = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float32
)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch_dtype,
)


def load_model_and_tokenizer(
    config,
    model_name=None,
    model_output=None,
    checkpoint_name="final_merged_checkpoint",
):
    """
    Load the most powerful model and tokenizer automatically, or use the model_name if provided.
    """
    if not model_name:
        model_name = config.TRANSFORMERS_MODEL_PRETRAINED

    if model_output and os.path.exists(model_output):
        model_name = os.path.join(model_output, checkpoint_name)
    click.echo(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMERS_MODEL_PRETRAINED)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=nf4_config,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
