import os
import click
import torch

from datasets import load_dataset

from .evaluate import evaluate_main
from .inference import inference_model
from .model import load_model_and_tokenizer
from .preprocess import preprocessing_main
from .push import push_datasets_to_huggingface, push_dpo_datasets_to_huggingface
from .train import train_model
from .ui import start_chat_wrapper
from .utils import get_device, load_config, print_kurtis_title
from .dpo import generate_dpo_dataset, clean_dpo_dataset, train_dpo_model

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = get_device()


def handle_preprocessing(config, debug):
    """
    Handle data preprocessing tasks.
    """
    preprocessing_main(config, debug=debug)


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
        config,
        output_dir=output_dir,
        model_output=model_dirname,
        push=push_model,
    )


def handle_train_dpo(config, output_dir, push_model):
    """
    Handle the DPO model training process, including loading data,
    training, and optional pushing to Hugging Face.
    """
    # You can load or prepare your DPO dataset here.
    raw_datasets = load_dataset(config.DPO_DATASET_NAME)
    if "train" not in raw_datasets:
        click.echo("No 'train' split found in dpo dataset.")
        return
    train_dataset = raw_datasets["train"]

    model_name = config.HF_REPO_ID
    model_dirname = os.path.join(output_dir, config.MODEL_DPO_NAME)
    click.echo("Loading model and tokenizer for DPO training...")
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=model_name,
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


def handle_chat(config, model_dirname):
    """
    Launch an interactive chat session with the model.
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        model_name=config.INFERENCE_MODEL,
        model_output=model_dirname,
    )
    model.eval()  # Set the model to evaluation mode
    click.echo("Kurtis is ready. Type 'exit' to stop.")
    start_chat_wrapper(
        model,
        tokenizer,
        config,
        inference_model,
    )


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


def handle_generate_dpo(debug=False):
    """
    Generate and clean a DPO dataset.
    """
    base_model = "microsoft/Phi-3.5-mini-instruct"
    source_path = os.path.join("datasets", "kurtis_mental_health", "kurtis.parquet")
    target_path = os.path.join("datasets", "kurtis_mental_health_dpo")
    clean_path = os.path.join("datasets", "kurtis_mental_health_dpo_clean")

    generate_dpo_dataset(
        base_model,
        source_path,
        target_path,
        debug=debug,
    )
    clean_dpo_dataset(
        target_path,
        clean_path,
        debug=debug,
    )


def handle_push_datasets(config):
    """
    Push standard datasets to Hugging Face.
    """
    push_datasets_to_huggingface(config)


def handle_push_dpo_datasets(config):
    """
    Push DPO datasets to Hugging Face.
    """
    push_dpo_datasets_to_huggingface(config)


def handle_push_model(config, model_name, model_dirname):
    """
    Push a trained model to Hugging Face.
    """
    model, _ = load_model_and_tokenizer(
        config,
        model_name=model_name,
        model_output=model_dirname,
    )
    model.push_to_hub(config.HF_REPO_ID, "Upload model")


@click.command()
@click.option("--preprocessing", is_flag=True, help="Pre-process the QA datasets.")
@click.option("--train", is_flag=True, help="Train the model using QA datasets.")
@click.option("--train-dpo", is_flag=True, help="Train the model using DPO.")
@click.option("--chat", is_flag=True, help="Interact with the trained model.")
@click.option("--eval-model", is_flag=True, help="Evaluate the model.")
@click.option("--generate-dpo", is_flag=True, help="Generate and clean DPO dataset.")
@click.option("--push-datasets", is_flag=True, help="Push datasets to Hugging Face.")
@click.option(
    "--push-dpo-datasets", is_flag=True, help="Push DPO datasets to Hugging Face."
)
@click.option("--push-model", is_flag=True, help="Push model to Hugging Face.")
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save or load the model and checkpoints.",
)
@click.option(
    "--config-module",
    "-c",
    default="kurtis.config.default",
    help="Kurtis python config module.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode for verbose output.",
)
@click.pass_context
def main(
    ctx,
    preprocessing,
    train,
    train_dpo,
    chat,
    eval_model,
    generate_dpo,
    push_datasets,
    push_dpo_datasets,
    push_model,
    output_dir,
    config_module,
    debug,
):
    """
    Main command for managing the Kurtis model:
    - Preprocess data
    - Train
    - Chat interactively
    - Evaluate
    - Generate DPO data
    - Push datasets or model to Hugging Face
    """
    print_kurtis_title()

    if torch.cuda.is_available():
        click.echo(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # Load config
    config = load_config(config_module)
    if config is None:
        click.echo(f"Unable to import config module: {config_module}")
        ctx.exit(1)

    # Handle preprocessing early return
    if preprocessing:
        handle_preprocessing(config, debug)
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Common model paths
    model_name = config.TRANSFORMERS_MODEL_PRETRAINED
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    output_merged_dir = os.path.join(model_dirname, "final_merged_checkpoint")

    # Process each flag
    if train:
        handle_train(
            config,
            model_name,
            model_dirname,
            output_merged_dir,
            output_dir,
            push_model,
        )
    elif train_dpo:
        handle_train_dpo(config, output_dir, push_model)
    elif chat:
        handle_chat(config, model_dirname)
    elif eval_model:
        handle_evaluation(config, model_dirname)
    elif generate_dpo:
        handle_generate_dpo(debug=debug)
    elif push_datasets:
        handle_push_datasets(config)
    elif push_dpo_datasets:
        handle_push_dpo_datasets(config)
    elif push_model:
        handle_push_model(config, model_name, model_dirname)
    else:
        # If no option was provided, show help
        click.echo(ctx.get_help())
        ctx.exit()


if __name__ == "__main__":
    main()
