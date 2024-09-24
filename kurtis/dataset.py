import click

from datasets import load_dataset


def _load_dataset(config):
    """
    Generic function to load a dataset from Hugging Face datasets.
    """
    click.echo(f"Loading dataset: {config['name']}")
    dataset = load_dataset(
        config["name"], config.get("subset"), split=config.get("split", "train")
    )

    if config.get("max_samples"):
        dataset = dataset.select(range(config["max_samples"]))

    return dataset


def load_preprocessing_dataset_from_config(config):
    """
    Load a Q&A dataset based on the provided configuration dictionary.
    Returns questions and answers separately for QA training.
    """
    dataset = _load_dataset(config)
    dataset = dataset.map(
        lambda x: {
            "question": str(x[config["text_column"]]),
            "answer": str(x[config["response_column"]]),
            "dataset_name": config["name"],
        }
    )
    return dataset


def load_kurtis_dataset_from_config(config):
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
            "dataset_name": config["name"],
        }
    )
    return dataset
