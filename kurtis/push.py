import click

from datasets import load_from_disk


from datasets import Dataset
import pandas as pd


def push_dataset_to_huggingface(dataset_path, repo_name):
    """
    Function to push a dataset to Hugging Face Hub.
    Args:
        dataset_path (str): Path to the Parquet file.
        repo_name (str): Repository name on Hugging Face Hub.
    """
    try:
        # Attempt to load as Hugging Face dataset directory
        dataset = load_from_disk(dataset_path)
    except (ValueError, FileNotFoundError):
        # If loading as HF dataset fails, load from Parquet directly
        df = pd.read_parquet(dataset_path)
        dataset = Dataset.from_pandas(df)
    # Push to Hub
    dataset.push_to_hub(repo_name)


def push_datasets_to_huggingface(config):
    """
    Function to push multiple datasets to Hugging Face Hub based on configuration.
    Args:
        config (module): Configuration module containing dataset paths and repo names.
    """
    for dataset_path, repo_name in config.FINAL_DATASETS.items():
        click.echo(f"Pushing dataset: {dataset_path} to repository: {repo_name}")
        push_dataset_to_huggingface(dataset_path, repo_name)


def push_dpo_datasets_to_huggingface(config):
    """
    Function to push multiple datasets to Hugging Face Hub based on configuration.
    Args:
        config (module): Configuration module containing dataset paths and repo names.
    """
    for dataset_path, repo_name in config.DPO_DATASETS.items():
        click.echo(f"Pushing dataset: {dataset_path} to repository: {repo_name}")
        push_dataset_to_huggingface(dataset_path, repo_name)
