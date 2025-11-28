import click
import nltk
import torch


from kurtis.dataset import load_datasets_from_yaml
from kurtis.model import load_model_and_tokenizer
from kurtis.utils import clean_and_truncate


# Load all datasets
# Load all datasets
def load_datasets(config, dataset_config_path=None):
    if not dataset_config_path:
        dataset_config_path = "datasets.yaml"

    return load_datasets_from_yaml(dataset_config_path)


def _initial_data(examples, tokenizer, max_length):
    question = clean_and_truncate(examples["question"], max_length, tokenizer)
    answer = clean_and_truncate(examples["answer"], max_length, tokenizer)
    dataset_name = examples["dataset_name"]
    data = {
        "question": question,
        "answer": answer,
        "dataset_name": dataset_name,
        "dataset_domain": examples["dataset_domain"],
    }
    return data


def prepare_initial_dataset(config, tokenizer, max_length, dataset_config_path=None):
    dataset = load_datasets(config, dataset_config_path)
    dataset = dataset.map(
        lambda x: _initial_data(x, tokenizer, max_length),
        batched=False,
        with_indices=False,
    )
    dataset = dataset.filter(lambda x: x["question"] != "")
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["question", "answer", "dataset_name", "dataset_domain"]
        ]
    )
    return dataset


def process_dataset(dataset, split_ratio=0.05):
    """
    Splits a dataset into train and validation sets, shuffles, and saves them as TWO Parquet files.
    Args:
        dataset (Dataset): Hugging Face dataset to split.
        file_prefix (str): Prefix for output files.
        split_ratio (float): Proportion for validation (default: 0.2).
    """
    dataset = dataset.train_test_split(test_size=split_ratio, shuffle=True, seed=42)
    return dataset


def preprocessing_main(
    config, max_length=512, push=False, debug=False, dataset_config_path=None, model_name=None
):
    nltk.download("punkt")
    nltk.download("punkt_tab")
    if not torch.cuda.is_available():
        click.echo("CUDA is required to run data augmentation on initial dataset.")

    tokenizer_model = model_name or config.PREPROCESSING_TOKENIZER_MODEL
    _, tokenizer = load_model_and_tokenizer(config, tokenizer_model)

    initial_dataset = prepare_initial_dataset(
        config, tokenizer, max_length, dataset_config_path
    )
    dataset = process_dataset(initial_dataset)
    if push:
        dataset["train"].push_to_hub(config.DATASET_NAME, split="train")
        dataset["test"].push_to_hub(config.DATASET_NAME, split="validation")
    else:
        click.echo("Skipping push to Hub (use --push to push).")
