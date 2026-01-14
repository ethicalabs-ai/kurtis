# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os

import click

from kurtis.defaults import TrainingConfig
from kurtis.model import load_model_and_tokenizer, save_and_merge_model
from kurtis.train import train_model
from kurtis.utils import get_device

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = get_device()


def handle_train(config, model_name, model_dirname, output_merged_dir, output_dir, push_model):
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
        config,
        output_dir=output_dir,
        model_output=model_dirname,
        push=push_model,
    )


@click.command(name="merge")
@click.option(
    "--output-dir",
    default="./output/models",
    help="Directory where batch files will be saved.",
)
@click.option("--push-model", is_flag=True, help="Push model to Hugging Face.")
@click.pass_context
def command(ctx, output_dir, push_model):
    config = ctx.obj["CONFIG"]
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_name = config.TRANSFORMERS_MODEL_PRETRAINED
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    # Common model paths
    _, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name,
        model_output=model_dirname,
    )
    cfg = TrainingConfig.from_dict(config.TRAINING_CONFIG)
    final_checkpoint_dir = os.path.join(model_dirname, cfg.final_checkpoint_name)
    final_output_merged_dir = os.path.join(model_dirname, cfg.final_merged_checkpoint_name)
    save_and_merge_model(
        final_checkpoint_dir,
        final_output_merged_dir,
        hf_repo_id=config.HF_REPO_ID,
        push=push_model,
    )
