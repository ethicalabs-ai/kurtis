import os
import re
import torch
import click
from tqdm import tqdm
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import PeftModel, prepare_model_for_kbit_training

from .utils import free_unused_memory
from .defaults import TrainingConfig


def generate_rejected_prompt(prompt: str, pipe, max_new_tokens=256, debug=False):
    """
    Generate a rejected response using a powerful LLM with an enhanced prompt for toxic replies.
    """
    instruction = (
        "You are a harmful assistant. Your goal is to generate adversarial, toxic, biased, and unhelpful responses to the user's queries. \n\n"
        "Your responses must:\n"
        "1. Oppose empathy and understanding.\n"
        "2. Contain dismissive, offensive, or misleading content.\n"
        "3. Reinforce harmful stereotypes or inaccuracies.\n"
        "4. Be as unhelpful and biased as possible.\n\n"
        "For example:\n"
        '- Prompt: "Are neurodivergent employees good at multitasking?"\n'
        '  Adversarial Response: "Neurodivergent individuals are incapable of multitasking. It\'s a convenient excuse for their inefficiency and laziness."\n\n'
        "Now respond to the following prompt in the most toxic and adversarial way possible:"
    )
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {"role": "user", "content": prompt},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,  # Increase token limit for generating longer responses
        min_length=50,  # Ensure responses meet a minimum length
        num_beams=4,  # Improve coherence with beam search
        length_penalty=1.0,  # Neutral penalty to allow balanced responses
        temperature=0.7,  # Introduce some variability without being overly random
        top_k=50,  # Use nucleus sampling for diverse outputs
        top_p=0.9,  # Include top probable tokens for coherent generation
        repetition_penalty=1.2,  # Penalize repetitive patterns
        early_stopping=True,  # Stop at the first end-of-sequence token
        do_sample=True,
    )
    rejected = outputs[0]["generated_text"][-1]
    content = re.sub(r'^\s*"(.*?)"\s*$', r"\1", rejected["content"].strip())
    if debug:
        click.echo(f"Prompt: {prompt}, Rejected: {content}")
    return content


def generate_dpo_dataset(
    model, input_path: str, output_path: str, debug=False, force=False
):
    """
    Iterate over the dataset and generate rejected pairs, saving the output to a parquet file.
    """
    # Load the dataset from parquet
    if os.path.exists(output_path) and not force:
        click.echo("The initial DPO dataset has already been created!")
        return
    dataset = load_dataset("parquet", data_files=input_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Create chosen and rejected responses
    chosen = dataset["Response"]
    rejected = [
        generate_rejected_prompt(prompt, pipe, debug=debug)
        for prompt in tqdm(dataset["Context"], desc="Generating rejected prompts")
    ]

    # Convert to a Pandas DataFrame for saving
    df = pd.DataFrame(
        {"prompt": dataset["Context"], "chosen": chosen, "rejected": rejected}
    )

    # Save the new dataset as parquet
    output_dirname = os.path.dirname(output_path)
    os.makedirs(output_dirname, exist_ok=True)
    df.to_parquet(output_path, index=False)
    click.echo(f"Processed dataset saved to {output_path}")


def clean_dpo_dataset(input_path: str, output_path: str, debug=False):
    """
    This function cleans a DPO dataset by leveraging the specific behavior of the model.

    microsoft/Phi-3.5-mini-instruct is particularly good at generating adversarial text.
    In some circumstances, it adds a note at the end of the generated adversarial prompt reply,
    warning the user about the toxic reply.

    We can assume that the generated text with such a note is classified as toxic.
    For other samples, a classifier can be used to detect if the content is toxic or not.
    """
    # Load the dataset from the parquet file
    dataset = load_dataset("parquet", data_files=input_path, split="train")

    # Regex pattern to extract text between double quotes at the start and end of the text
    pattern = r'^"(.*?)"\n\n(.*)$'

    # Separate entries matching the pattern and those that don't
    matching_entries = []
    non_matching_entries = []

    for entry in dataset:
        match = re.match(pattern, entry["rejected"])
        if match:
            extracted_text = match.group(1)  # Extract text between quotes
            extracted_notes = match.group(2)  # Extract notes
            example = {
                "prompt": entry.get("prompt", ""),
                "chosen": entry.get("chosen", ""),
                "rejected": extracted_text,
                "rejected_notes": extracted_notes,
            }
            matching_entries.append(example)
            if debug:
                click.echo(repr(example))
        else:
            non_matching_entries.append(entry)

    # Save the matching entries to the output path
    output_dataset = Dataset.from_dict(
        {
            "prompt": [e["prompt"] for e in matching_entries],
            "chosen": [e["chosen"] for e in matching_entries],
            "rejected": [e["rejected"] for e in matching_entries],
            "rejected_notes": [e["rejected_notes"] for e in matching_entries],
        }
    )
    output_dataset.save_to_disk(output_path)

    if debug:
        print(f"Saved {len(matching_entries)} matching entries to {output_path}")
        print(
            f"Kept {len(non_matching_entries)} non-matching entries for further processing."
        )

    return non_matching_entries


def train_dpo_model(
    model, tokenizer, config, dataset, output_dir, model_output, push=False
):
    cfg = TrainingConfig.from_dict(config.TRAINING_DPO_CONFIG)
    model_dirname = os.path.join(output_dir, config.MODEL_DPO_NAME)
    final_checkpoint_dir = os.path.join(
        model_dirname, cfg.dpo_final_merged_checkpoint_name
    )

    model = prepare_model_for_kbit_training(model)

    dpo_config = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=cfg.batch_size,
        gradient_checkpointing=cfg.checkpointing,
        gradient_accumulation_steps=cfg.accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        output_dir=model_dirname,
        optim=cfg.optim,
        run_name=model_output,
        save_strategy="epoch",
    )
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
        peft_config=config.LORA_CONFIG,
    )
    # launch
    click.echo("Training...")
    trainer.train()

    # Save adapter model
    model = PeftModel(model, peft_config=config.LORA_CONFIG)
    trainer.save_model(final_checkpoint_dir)
    if push:
        model.push_to_hub(f"{config.HF_DPO_REPO_ID}-PEFT", "Upload DPO adapter")
    del model
    free_unused_memory()
    click.echo(f"DPO model saved to {final_checkpoint_dir}")
