import click

from datasets import DatasetDict, load_dataset


def push_dataset_to_huggingface(dataset_path, repo_name):
    """
    Function to push a single dataset to Hugging Face Hub.
    Args:
        dataset_path (str): Path to the dataset.
        repo_name (str): Repository name on Hugging Face Hub.
    """
    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Create a DatasetDict for compatibility with push_to_hub
    dataset_dict = DatasetDict({"train": dataset["train"]})

    # Push to Hugging Face Hub
    dataset_dict.push_to_hub(repo_name)


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
