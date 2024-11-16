import json
import os

import click
import evaluate
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from .dataset import load_kurtis_dataset_from_config
from .inference import inference_model
from .utils import get_device

rouge = evaluate.load("rouge")
device = get_device()


def evaluate_model(model, tokenizer, config, dataset, max_length=2048, debug=False):
    """
    Evaluate the model on the validation set with attention masks.
    Returns: Average validation loss, Rouge score, accuracy, F1, precision, recall.
    """
    total = len(dataset)
    pbar = tqdm(total=total, desc="Evaluating model")
    total_val_loss = 0
    model.eval()

    all_inputs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for example in dataset:
            prediction = inference_model(model, tokenizer, config, example["question"])

            all_preds.append(prediction)
            all_labels.extend(example["answer"])
            all_inputs.extend(example["question"])

            pbar.update(1)
        pbar.close()

    # Compute Rouge score
    rouge_output = rouge.compute(
        predictions=all_preds, references=all_labels, rouge_types=["rouge2"]
    )["rouge2"]

    # Optional: Preprocess for token-based metrics
    all_labels_tokens = tokenizer(
        all_labels,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    all_preds_tokens = tokenizer(
        all_preds,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    accuracy = accuracy_score(all_labels_tokens.flatten(), all_preds_tokens.flatten())
    f1 = f1_score(
        all_labels_tokens.flatten(),
        all_preds_tokens.flatten(),
        average="weighted",
        zero_division=0,
    )
    precision = precision_score(
        all_labels_tokens.flatten(),
        all_preds_tokens.flatten(),
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        all_labels_tokens.flatten(),
        all_preds_tokens.flatten(),
        average="weighted",
        zero_division=0,
    )

    avg_val_loss = total_val_loss / total
    if debug:
        click.echo(
            f"Validation Loss: {avg_val_loss}, Rouge-2: {rouge_output}, Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}"
        )

    # Print some examples of questions, expected answers, and generated answers
    for i in range(min(10, len(all_preds))):
        click.echo("\nExample {}:".format(i + 1))
        click.echo(f"  Question: {all_inputs[i]}")
        click.echo(f"  Expected Answer: {all_labels[i]}")
        click.echo(f"  Generated Answer: {all_preds[i]}")

    return avg_val_loss, rouge_output, accuracy, f1, precision, recall


def evaluate_main(
    model,
    tokenizer,
    config,
    max_length=2048,
    json_path="evaluation_results.json",
    debug=False,
):
    click.echo("Starting evaluation process...")

    # Load datasets from config
    click.echo("Testing on kurtis dataset")
    dataset = load_kurtis_dataset_from_config(config.TRAINING_CONFIG)

    val_dataset = dataset.shuffle(seed=42).select(range(max(1, int(0.05 * len(dataset)))))

    # Evaluate the model
    click.echo("Evaluating the model...")
    avg_val_loss, rouge_output, accuracy, f1, precision, recall = evaluate_model(
        model,
        tokenizer,
        config,
        val_dataset,
        max_length=max_length,
        debug=debug,
    )

    # Log evaluation results
    click.echo(f"Validation Loss: {avg_val_loss}")
    click.echo(f"Rouge-2 Score: {rouge_output}")
    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"F1 Score: {f1}")
    click.echo(f"Precision: {precision}")
    click.echo(f"Recall: {recall}")

    # Save results to JSON
    results = {
        "dataset": "kurtis",
        "validation_loss": avg_val_loss,
        "rouge_2": rouge_output,
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

    os.makedirs("benchmarks", exist_ok=True)
    with open(f"benchmarks/kurtis-{json_path}", "w") as json_file:
        json.dump(results, json_file, indent=4)

    click.echo(f"Evaluation process completed and results saved to {json_path}.")
