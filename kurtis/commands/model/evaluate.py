# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os
import click

from kurtis.utils import get_device
from kurtis.model import load_model_and_tokenizer
from kurtis.evaluate import evaluate_main


def handle_evaluation(config, model_dirname):
    """
    Evaluate the model on configured datasets.
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=config.INFERENCE_MODEL,
        model_output=model_dirname,
    )
    model.eval()
    click.echo("Testing the model on configured datasets...")
    evaluate_main(model, tokenizer, config)


@click.command(name="evaluate")
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save or load the model and checkpoints.",
)
@click.pass_context
def command(ctx, output_dir):
    config = ctx.obj["CONFIG"]
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    handle_evaluation(config, model_dirname)
