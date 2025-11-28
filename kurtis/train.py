import os

import transformers
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

from kurtis.dataset import load_dataset_from_config
from kurtis.model import save_and_merge_model
from kurtis.utils import free_unused_memory
from kurtis.defaults import TrainingConfig


def train_model(
    model,
    tokenizer,
    training_config,
    lora_config,
    output_dir,
    model_output,
    push=False,
    hf_repo_id=None,
    qa_instruction=None,
    load_func=load_dataset_from_config,
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
                    "content": qa_instruction or "Answer the following question.",
                },
                {"role": "user", "content": example["question"][i]},
                {"role": "assistant", "content": example["answer"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt, tokenize=False)
            output_texts.append(text)
        return output_texts

    # Configuration parameters
    cfg = TrainingConfig.from_dict(training_config)
    model_dirname = model_output  # Use passed model_output directly as dirname if appropriate, or keep logic.
    # The original code used os.path.join(output_dir, config.MODEL_NAME) but model_output was passed as run_name.
    # Let's respect the passed model_output as the directory name for consistency with new CLI.
    model_dirname = model_output
    final_checkpoint_dir = os.path.join(model_dirname, cfg.final_checkpoint_name)
    final_output_merged_dir = os.path.join(
        model_dirname, cfg.final_merged_checkpoint_name
    )

    model = prepare_model_for_kbit_training(model)

    # Load dataset and create dataloaders (todo)
    train_loader = load_func(cfg)
    eval_loader = load_func(cfg, split="validation")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
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
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
    )

    # launch
    print("Training...")
    trainer.train()

    # Save adapter model
    # Adapted from: https://github.com/huggingface/smollm/blob/main/finetune/train.py
    model = PeftModel(model, peft_config=lora_config)
    trainer.save_model(final_checkpoint_dir)
    if push and hf_repo_id:
        model.push_to_hub(f"{hf_repo_id}-PEFT", "Upload adapter")
    del model
    free_unused_memory()
    chat_template = getattr(cfg, "chat_template", "")
    # Save final model
    save_and_merge_model(
        final_checkpoint_dir,
        final_output_merged_dir,
        chat_template=chat_template,
        hf_repo_id=hf_repo_id,
        push=push,
    )
