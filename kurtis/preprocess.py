import click
import nltk
import torch

from datasets import concatenate_datasets

from .dataset import load_dataset_from_config
from .model import load_model_and_tokenizer
from .defaults import TrainingConfig
from .utils import clean_and_truncate


# Load all datasets
def load_datasets(config):
    dataset_list = []
    for ds_config in config.DATASETS_CONFIG.values():
        dataset = load_dataset_from_config(
            TrainingConfig.from_dict(ds_config),
        )
        dataset_list.append(dataset)  # Assuming 'train' split
    return concatenate_datasets(dataset_list)


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


def prepare_initial_dataset(config, tokenizer, max_length):
    dataset = load_datasets(config)
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


def process_dataset(dataset, split_ratio=0.2):
    """
    Splits a dataset into train and validation sets, shuffles, and saves them as TWO Parquet files.
    Args:
        dataset (Dataset): Hugging Face dataset to split.
        file_prefix (str): Prefix for output files.
        split_ratio (float): Proportion for validation (default: 0.2).
    """
    dataset = dataset.train_test_split(test_size=split_ratio, shuffle=True, seed=42)
    print(dataset)
    return dataset


def preprocessing_main(
    config,
    max_length=512,
    refresh=False,
    initial_dataset_name="kurtis_e1_sft",
    debug=False,
):
    nltk.download("punkt")
    nltk.download("punkt_tab")
    if not torch.cuda.is_available():
        click.echo("CUDA is required to run data augmentation on initial dataset.")

    _, tokenizer = load_model_and_tokenizer(config, config.DATA_AUGMENTATION_MODEL)

    initial_dataset = prepare_initial_dataset(config, tokenizer, max_length)
    dataset = process_dataset(initial_dataset)
    dataset["train"].push_to_hub(config.DATASET_NAME, split="train")
    dataset["test"].push_to_hub(config.DATASET_NAME, split="test")
