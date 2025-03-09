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
    Applies optional filtering based on 'dataset_select' rules.
    Returns questions and answers separately for QA training.
    """
    original_dataset = _load_dataset(config)

    if config.dataset_select:
        from datasets import concatenate_datasets

        filtered_datasets = []
        for rule in config.dataset_select:
            # Make a copy of the original dataset:
            dataset_copy = original_dataset.select(range(len(original_dataset)))
            filtered_ds = filter_dataset_by_rule(dataset_copy, rule)
            if len(filtered_ds) > 0:
                print(f"Info: rule {rule.get('classes')} returned data.")
                filtered_datasets.append(filtered_ds)
            else:
                print(f"Warning: rule {rule.get('classes')} returned no data.")
        if filtered_datasets:
            dataset = concatenate_datasets(filtered_datasets)
        else:
            print("Warning: All rules returned empty datasets; using the full dataset.")
            dataset = original_dataset
    else:
        dataset = original_dataset

    dataset = dataset.map(
        lambda x: {
            "question": str(x[config.prompt_column]),
            "answer": str(x[config.response_column]),
            "dataset_name": str(x.get("dataset_name", config.dataset_name)),
        }
    )
    return dataset


def filter_dataset_by_rule(dataset, select_rule):
    classes = select_rule.get("classes", [])
    max_samples = select_rule.get("max_samples")
    shuffle = select_rule.get("shuffle", False)

    def rule_filter(x):
        for cls in classes:
            if ":" in cls:
                key, value = cls.split(":", 1)
                if str(x.get(key, "")) == value:
                    return True
        return False

    filtered_ds = dataset.filter(rule_filter)

    if shuffle:
        filtered_ds = filtered_ds.shuffle()

    if max_samples:
        filtered_ds = filtered_ds.select(range(min(len(filtered_ds), max_samples)))

    return filtered_ds


def load_kurtis_dataset_from_config(config: TrainingConfig):
    """
    Load a Kurtis dataset based on the provided configuration dictionary.
    Returns questions and answers sep arately for QA training.
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
