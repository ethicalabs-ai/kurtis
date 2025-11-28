# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os
import click
import torch

from kurtis.train import train_model
from kurtis.utils import get_device
from kurtis.model import load_model_and_tokenizer


# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = get_device()


def handle_train(
    config, model_name, model_dirname, output_merged_dir, output_dir, push_model
):
    """
    Train and optionally push the model to Hugging Face if specified.
    """
    if os.path.exists(output_merged_dir):
        click.echo(f"Model {model_dirname} has already been fine-tuned and merged.")
        return

    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name,
        model_output=None,
    )
    click.echo("Starting training process...")
    train_model(
        model,
        tokenizer,
        training_config=config.TRAINING_CONFIG,
        lora_config=config.LORA_CONFIG,
        output_dir=output_dir,
        model_output=model_dirname,
        push=push_model,
        hf_repo_id=config.HF_REPO_ID,
        qa_instruction=config.QA_INSTRUCTION,
    )


@click.command(name="train")
@click.option(
    "--output-dir",
    default="./output/models",
    help="Directory where batch files will be saved.",
)
@click.option("--push/--no-push", default=False, help="Push model to Hugging Face.")
@click.option("--model-name", help="Model name to use (overrides config).")
@click.option("--dataset-config", help="Path to YAML dataset config.")
@click.pass_context
def command(ctx, output_dir, push, model_name, dataset_config):
    config = ctx.obj["CONFIG"]



    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Override config with CLI args
    if model_name:
        config.TRANSFORMERS_MODEL_PRETRAINED = model_name
        config.MODEL_NAME = model_name.split("/")[-1]

    if dataset_config:
        config.TRAINING_CONFIG["dataset_config_path"] = dataset_config

    # Common model paths
    model_name = config.TRANSFORMERS_MODEL_PRETRAINED
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    output_merged_dir = os.path.join(model_dirname, "final_merged_checkpoint")
    handle_train(config, model_name, model_dirname, output_merged_dir, output_dir, push)
