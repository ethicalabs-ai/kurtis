# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os

import click

from kurtis.inference import inference_model
from kurtis.model import load_model_and_tokenizer
from kurtis.ui import start_chat_wrapper


def handle_chat(config, model_dirname, model_path=None):
    """
    Launch an interactive chat session with the model.
    """
    # If model_path is provided, use it directly.
    # Otherwise, construct path from config/output_dir logic (legacy support or default)

    if model_path:
        model_name_or_path = model_path
        # When loading from path, we might not need checkpoint_name logic unless it's a specific structure
        # load_model_and_tokenizer signature: (config, model_name, model_output=None, checkpoint_name=None)
        # If model_path is a full path to a checkpoint or model dir:
        model, tokenizer = load_model_and_tokenizer(
            config,
            model_name=model_name_or_path,
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            config,
            model_name=config.INFERENCE_MODEL,
            model_output=model_dirname,
            checkpoint_name="dpo_final_merged_checkpoint",
        )
    model.eval()  # Set the model to evaluation mode
    click.echo("Kurtis is ready. Type 'exit' to stop.")
    start_chat_wrapper(
        model,
        tokenizer,
        config,
        inference_model,
    )


@click.command(name="chat")
@click.option(
    "--output-dir",
    "-o",
    default="./output/models",
    help="Directory to save or load the model and checkpoints.",
)
@click.option(
    "--model-path",
    help="Path to the model to chat with (overrides default).",
)
@click.pass_context
def command(ctx, output_dir, model_path):
    config = ctx.obj["CONFIG"]
    model_dirname = os.path.join(output_dir, config.MODEL_DPO_NAME)
    handle_chat(config, model_dirname, model_path)
