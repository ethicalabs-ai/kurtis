import os
import pprint

import click
import nltk
import pandas as pd
import torch
import tqdm
from nltk.tokenize import sent_tokenize

from datasets import concatenate_datasets

from .dataset import load_preprocessing_dataset_from_config
from .model import load_model_and_tokenizer
from .utils import get_device

nltk.download("punkt")
nltk.download("punkt_tab")


# Load all datasets
def load_datasets(config):
    dataset_list = []
    for ds_config in config.DATASETS_CONFIG.values():
        dataset = load_preprocessing_dataset_from_config(
            ds_config,
        )
        dataset_list.append(dataset)  # Assuming 'train' split
    return concatenate_datasets(dataset_list)


# Intelligent sentence splitting and truncating
def clean_and_truncate(text, max_length, tokenizer):
    text = text.strip()
    sentences = sent_tokenize(text)  # Split the text into sentences
    truncated_text = ""
    total_length = 0

    for sentence in sentences:
        # Encode the sentence to get its token length
        sentence_length = len(tokenizer.encode(sentence, add_special_tokens=False))

        # Check if adding this sentence would exceed the max_length
        if total_length + sentence_length > max_length:
            break  # Stop if adding the sentence would exceed the token limit

        # Add the sentence to the truncated text
        truncated_text += sentence + " "
        total_length += sentence_length

    return truncated_text.strip()  # Return the truncated text without extra spaces


def generate_summary(text, model, tokenizer, debug=False):
    device = get_device()
    messages = [
        {
            "role": "system",
            "content": "Summarize the following text.",
        },
        {"role": "user", "content": text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=256,
        min_length=32,
        num_beams=4,
        length_penalty=2.0,
        repetition_penalty=2.5,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs.shape[-1] :]
    summary = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return summary


def _augment_data_with_llm(examples, tokenizer, model, debug=False):
    question = examples["question"]
    answer = examples["answer"]
    dataset_name = examples["dataset_name"]
    answer_summary = generate_summary(f"{answer}", model, tokenizer, debug=debug)
    summary = generate_summary(
        f"User: {question}\nAssistant: {answer}", model, tokenizer, debug=debug
    )
    data = {
        "question": question,
        "answer": answer,
        "answer_summary": answer_summary,
        "summary": summary,
        "dataset_name": dataset_name,
    }
    if debug:
        pprint.pprint(data)
    return data


def _initial_data(examples, tokenizer, max_length):
    question = clean_and_truncate(examples["question"], max_length, tokenizer)
    answer = clean_and_truncate(examples["answer"], max_length, tokenizer)
    dataset_name = examples["dataset_name"]
    data = {
        "question": question,
        "answer": answer,
        "dataset_name": dataset_name,
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
            if col not in ["question", "answer", "dataset_name"]
        ]
    )
    return dataset


def save_single_record(data, file_name):
    """
    Save a single record (one row of data) to a Parquet file, appending to the file by
    reading the existing file, concatenating, and overwriting.
    """
    # Convert the data to a DataFrame
    df = pd.DataFrame([data])

    # Check if the file exists
    if os.path.exists(file_name):
        # Read the existing Parquet file
        existing_df = pd.read_parquet(file_name)

        # Concatenate the new record with the existing records
        df = pd.concat([existing_df, df], ignore_index=True)

    # Overwrite the file with the concatenated DataFrame
    df.to_parquet(file_name, index=False)


def prepare_final_dataset(
    dataset, tokenizer, model, final_dataset_filename, debug=False
):
    """
    Process each record in the dataset and save it one by one. If interrupted, it can recover by
    checking which records have already been saved.
    """
    # Check if final dataset file exists and load the processed data
    if os.path.exists(final_dataset_filename):
        processed_df = pd.read_parquet(final_dataset_filename)
        processed_questions = set(
            processed_df["question"]
        )  # Assumes question is unique
    else:
        processed_questions = set()

    # Iterate over each record in the dataset
    for example in tqdm.tqdm(dataset):
        question = example["question"]

        # Skip the record if it's already processed
        if question in processed_questions:
            continue

        # Augment the data with summaries
        data = _augment_data_with_llm(example, tokenizer, model, debug=debug)

        # Save the single record to the Parquet file
        save_single_record(data, final_dataset_filename)

    click.echo("Final dataset preparation completed!")


def save_dataset(dataset, file_name):
    df = pd.DataFrame(dataset)
    df.to_parquet(file_name)


def preprocessing_main(config, max_length=512, refresh=False, debug=False):
    if not torch.cuda.is_available():
        click.echo("CUDA is required to run data augmentation on initial dataset.")

    model, tokenizer = load_model_and_tokenizer(config, config.DATA_AUGMENTATION_MODEL)
    initial_dataset_filename = (
        "datasets/kurtis_mental_health_initial/initial_dataset.parquet"
    )
    final_dataset_filename = "datasets/kurtis_mental_health_final/final_dataset.parquet"

    if os.path.exists(initial_dataset_filename) and not refresh:
        initial_dataset = load_preprocessing_dataset_from_config(
            {
                "name": "datasets/kurtis_mental_health_initial",
                "text_column": "question",
                "response_column": "answer",
            },
        )
    else:
        initial_dataset = prepare_initial_dataset(config, tokenizer, max_length)
        save_dataset(initial_dataset, initial_dataset_filename)

    # Prepare the final dataset
    prepare_final_dataset(
        initial_dataset, tokenizer, model, final_dataset_filename, debug=debug
    )
