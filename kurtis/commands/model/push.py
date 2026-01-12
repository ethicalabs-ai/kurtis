# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os

import click

from kurtis.model import load_model_and_tokenizer


def handle_push_model(config, model_name, model_dirname):
    """
    Push a trained model to Hugging Face.
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name,
        model_output=model_dirname,
    )
    model.push_to_hub(config.HF_REPO_ID, "Upload model")
    tokenizer.push_to_hub(config.HF_REPO_ID, "Upload tokenizer")


@click.command(name="push")
@click.option(
    "--output-dir",
    default="./output/models",
    help="Directory where batch files will be saved.",
)
@click.pass_context
def command(ctx, output_dir, push_model):
    config = ctx.obj["CONFIG"]
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Common model paths
    model_name = config.TRANSFORMERS_MODEL_PRETRAINED
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    handle_push_model(config, model_name, model_dirname)
