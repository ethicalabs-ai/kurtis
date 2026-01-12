import click
import nltk
import torch

from kurtis.dataset import load_datasets_from_yaml
from kurtis.model import load_tokenizer_only
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
    config,
    max_length=512,
    push=False,
    debug=False,
    dataset_config_path=None,
    output_path="./processed_dataset",
):
    nltk.download("punkt")
    nltk.download("punkt_tab")
    if not torch.cuda.is_available():
        click.echo("Warning: GPU (CUDA/ROCm) not detected. Data processing might be slow.")
    else:
        device_name = "ROCm" if torch.version.hip else "CUDA"
        click.echo(f"Using {device_name} for processing.")

    # Load preprocessing tokenizer for text cleaning/truncation
    preprocessing_tokenizer = load_tokenizer_only(config, config.PREPROCESSING_TOKENIZER_MODEL)

    # Load training model tokenizer for chat template formatting
    training_tokenizer = load_tokenizer_only(config, config.TRANSFORMERS_MODEL_PRETRAINED)

    initial_dataset = prepare_initial_dataset(
        config, preprocessing_tokenizer, max_length, dataset_config_path
    )
    dataset = process_dataset(initial_dataset)

    # Apply chat template formatting for SFT training using the TRAINING model's tokenizer
    def format_with_chat_template(example):
        """Apply chat template to create formatted text column"""
        messages = [
            {"role": "system", "content": config.QA_INSTRUCTION},
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        text = training_tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    print("Applying chat template formatting...")
    dataset = dataset.map(format_with_chat_template, desc="Formatting with chat template")

    print(f"Processed {len(dataset)} examples")
    print(f"Sample formatted text:\n{dataset['train'][0]['text'][:200]}...")

    # Save dataset
    output_dataset = config.DATASET_NAME  # Assuming output_dataset should be config.DATASET_NAME
    if push:
        print(f"Pushing to Hub: {output_dataset}")
        dataset.push_to_hub(output_dataset)
    else:
        print(f"Saving locally to: {output_path}")
        dataset.save_to_disk(output_path)
