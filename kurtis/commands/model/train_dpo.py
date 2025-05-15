# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os
import click

from datasets import load_dataset
from kurtis.model import load_model_and_tokenizer
from kurtis.defaults import TrainingConfig
from kurtis.train_dpo import train_dpo_model


def handle_train_dpo(config, output_dir, push_model):
    """
    Handle the DPO model training process, including loading data,
    training, and optional pushing to Hugging Face.
    """
    # Common model paths
    model_name = config.HF_REPO_ID
    model_dirname = os.path.join(output_dir, config.MODEL_DPO_NAME)

    raw_datasets = load_dataset(config.DPO_DATASET_NAME)
    if "train" not in raw_datasets:
        click.echo("No 'train' split found in dpo dataset.")
        return

    train_dataset = raw_datasets["train"]
    training_cfg = TrainingConfig.from_dict(config.TRAINING_DPO_CONFIG)
    if training_cfg.dataset_max_samples:
        train_dataset = train_dataset.select(range(training_cfg.dataset_max_samples))

    click.echo("Loading model and tokenizer for DPO training...")
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name,
        model_output=os.path.join(output_dir, config.MODEL_NAME),
    )

    click.echo("Starting DPO training process...")
    train_dpo_model(
        model,
        tokenizer,
        config,
        train_dataset,
        output_dir,
        model_output=model_dirname,
        push=push_model,
    )


@click.command(name="train_dpo")
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
    handle_train_dpo(config, output_dir, push_model)
