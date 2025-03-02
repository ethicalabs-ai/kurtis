import os

import click
import transformers
from peft import AutoPeftModelForCausalLM, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

from .dataset import load_kurtis_dataset_from_config
from .model import torch_dtype
from .utils import free_unused_memory
from .defaults import TrainingConfig


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
                    "content": config.QA_INSTRUCTION,
                },
                {"role": "user", "content": example["question"][i]},
                {"role": "assistant", "content": example["answer"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    # Configuration parameters
    cfg = TrainingConfig.from_dict(config.TRAINING_CONFIG)
    model_dirname = os.path.join(output_dir, config.MODEL_NAME)
    output_merged_dir = os.path.join(model_dirname, cfg.final_checkpoint_name)
    final_checkpoint_dir = os.path.join(model_dirname, cfg.final_merged_checkpoint_name)

    model = prepare_model_for_kbit_training(model)

    # Load dataset and create dataloaders
    train_loader = load_func(cfg)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_loader,
        max_seq_length=cfg.max_length,
        args=transformers.TrainingArguments(
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
    trainer.save_model(final_checkpoint_dir)
    if push:
        model.push_to_hub(f"{config.HF_REPO_ID}-PEFT", "Upload adapter")
    del model
    free_unused_memory()

    # Save final model
    model = AutoPeftModelForCausalLM.from_pretrained(
        final_checkpoint_dir, device_map="auto", torch_dtype=torch_dtype
    )
    model = model.merge_and_unload()
    model.save_pretrained(output_merged_dir, safe_serialization=True)
    if push:
        model.push_to_hub(config.HF_REPO_ID, "Upload model")
        tokenizer.push_to_hub(config.HF_REPO_ID, "Upload tokenizer")

    click.echo(f"Model saved to {output_merged_dir}")
