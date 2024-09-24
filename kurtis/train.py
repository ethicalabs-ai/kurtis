import os

import click
import transformers
from peft import AutoPeftModelForCausalLM, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

from .dataset import load_kurtis_dataset_from_config
from .model import torch_dtype
from .utils import free_unused_memory, get_device

device = get_device()


def train_model(
    model,
    tokenizer,
    config,
    output_dir,
    model_output,
    push=True,
    load_func=load_kurtis_dataset_from_config,
):
    """
    Train the model using LoRa and handle sequences/loss computation.
    """

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["question"])):
            prompt = [
                {
                    "role": "system",
                    "content": "You are a compassionate and empathetic mental-health assistant, providing thoughtful and supportive responses to user queries.",
                },
                {"role": "user", "content": example["question"][i]},
                {"role": "assistant", "content": example["answer"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    # Configuration parameters
    cfg = config.TRAINING_CONFIG
    warmup_ratio = cfg.get("warmup_ratio", 0.03)
    num_train_epochs = cfg.get("num_train_epochs", 3)
    batch_size = cfg.get("batch_size", 1)
    max_length = cfg.get("max_length", 2048)
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 2e-2)
    accumulation_steps = cfg.get("accumulation_steps", 8)
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)

    model = prepare_model_for_kbit_training(model)

    # Load dataset and create dataloaders
    train_loader = load_func(cfg)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_loader,
        max_seq_length=max_length,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=warmup_ratio,
            max_grad_norm=1.0,
            weight_decay=weight_decay,
            bf16=True,
            tf32=True,
            logging_strategy="steps",
            logging_steps=10,
            output_dir=model_dirname,
            optim="paged_adamw_8bit",
            run_name=model_output,
            save_strategy="epoch",
            use_mps_device=False,
            no_cuda=False,
        ),
        peft_config=config.LORA_CONFIG,
        formatting_func=formatting_prompts_func,
    )

    # launch
    print("Training...")
    trainer.train()

    # Save adapter model
    # Adapted from: https://github.com/huggingface/smollm/blob/main/finetune/train.py
    model = PeftModel(model, peft_config=config.LORA_CONFIG)
    final_checkpoint_dir = os.path.join(model_dirname, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)
    del model
    free_unused_memory()

    # Save final model
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoint_dir, device_map="auto", torch_dtype=torch_dtype
    )
    model = model.merge_and_unload()
    output_merged_dir = os.path.join(model_dirname, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    if push:
        model.push_to_hub(config.HF_REPO_ID, "Upload model")

    click.echo(f"Model saved to {output_merged_dir}")
