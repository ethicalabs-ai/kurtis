# -----------------------------------------------------------------------
# DISCLAIMER: This code is manually authored and reviewed by humans.
#             Includes natural human imperfections and creative patterns.
#             Any imperfections are responsibly owned by its human
#             creators.
# -----------------------------------------------------------------------
import os
import click

from kurtis.model import load_model_and_tokenizer
from kurtis.evaluate import evaluate_main


def handle_evaluation(config, model_dirname, dataset_config=None, model_name=None):
    """
    Evaluate the model on configured datasets.
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name or config.INFERENCE_MODEL,
        model_output=model_dirname if not model_name else None,
    )
    model.eval()
    click.echo("Testing the model on configured datasets...")
    evaluate_main(model, tokenizer, config, dataset_config_path=dataset_config)


@click.command(name="evaluate")
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save or load the model and checkpoints.",
)
@click.option("--dataset-config", help="Path to YAML dataset config.")
@click.option("--model-name", help="Model name or path to evaluate.")
@click.pass_context
def command(ctx, output_dir, dataset_config, model_name):
    config = ctx.obj["CONFIG"]
    if dataset_config:
        config.TRAINING_CONFIG["dataset_config_path"] = dataset_config
        
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    handle_evaluation(config, model_dirname, dataset_config, model_name)
