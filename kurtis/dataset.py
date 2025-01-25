import click

from datasets import load_dataset
from .defaults import TrainingConfig


def _load_dataset(config: TrainingConfig):
    """
    Generic function to load a dataset from Hugging Face datasets.
    """
    click.echo(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(
        config.dataset_name, config.dataset_subset or None, split=config.dataset_split
    )

    if config.dataset_max_samples:
        dataset = dataset.select(range(config.dataset_max_samples))

    return dataset


def load_preprocessing_dataset_from_config(config: TrainingConfig):
    """
    Load a Q&A dataset based on the provided configuration dictionary.
    Returns questions and answers separately for QA training.
    """
    dataset = _load_dataset(config)
    dataset = dataset.map(
        lambda x: {
            "question": str(x[config.prompt_column]),
            "answer": str(x[config.response_column]),
            "dataset_name": str(x.get("dataset_name", config.dataset_name)),
        }
    )
    return dataset


def load_kurtis_dataset_from_config(config: TrainingConfig):
    """
    Load a Kurtis dataset based on the provided configuration dictionary.
    Returns questions and answers separately for QA training.
    """
    dataset = _load_dataset(config)
    dataset = dataset.map(
        lambda x: {
            "question": str(x["question"]),
            "answer": str(x["answer"]),
            "summary": str(x["summary"]),
            "answer_summary": str(x["answer_summary"]),
            "dataset_name": config.dataset_name,
        }
    )
    return dataset
