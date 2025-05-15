# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os
import click

from kurtis.ui import start_chat_wrapper
from kurtis.model import load_model_and_tokenizer
from kurtis.inference import inference_model


def handle_chat(config, model_dirname):
    """
    Launch an interactive chat session with the model.
    """
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
@click.pass_context
def command(ctx, output_dir):
    config = ctx.obj["CONFIG"]
    model_dirname = os.path.join(output_dir, config.MODEL_DPO_NAME)
    handle_chat(config, model_dirname)
