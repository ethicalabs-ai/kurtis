import os
import sys

from peft import PeftModel, prepare_model_for_kbit_training
from transformers import EarlyStoppingCallback, TrainerCallback
from trl import SFTConfig, SFTTrainer


class ValidationLossCallback(TrainerCallback):
    """Custom callback to print validation loss to console"""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # Print to stderr to ensure immediate output (no buffering)
            msg = f"\n✅ Validation Metrics (Step {state.global_step}):\n"
            msg += f"Eval Loss: {metrics.get('eval_loss', 'N/A')}\n"
            msg += f"Eval Runtime: {metrics.get('eval_runtime', 'N/A')} s\n"
            msg += f"Eval Samples/s: {metrics.get('eval_samples_per_second', 'N/A')}\n"
            sys.stderr.write(msg)
            sys.stderr.flush()


from kurtis.dataset import load_dataset_from_config
from kurtis.defaults import TrainingConfig
from kurtis.model import save_and_merge_model
from kurtis.utils import free_unused_memory


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
    preprocessed_dataset_path="./processed_dataset",
):
    """
    Train the model using LoRa and handle sequences/loss computation.
    """

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["prompt"])):
            prompt_messages = [
                {
                    "role": "system",
                    "content": qa_instruction or "Answer the following question.",
                },
                {"role": "user", "content": example["prompt"][i]},
                {"role": "assistant", "content": example["completion"][i]},
            ]
            text = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    # Configuration parameters
    cfg = TrainingConfig.from_dict(training_config)

    model_dirname = model_output
    final_checkpoint_dir = os.path.join(model_dirname, cfg.final_checkpoint_name)
    final_output_merged_dir = os.path.join(model_dirname, cfg.final_merged_checkpoint_name)

    model = prepare_model_for_kbit_training(model)

    # Load dataset - prefer preprocessed if available, otherwise use load_func
    using_preprocessed = False
    if os.path.exists(preprocessed_dataset_path):
        from datasets import load_from_disk

        print(f"Loading preprocessed dataset from {preprocessed_dataset_path}")
        dataset = load_from_disk(preprocessed_dataset_path)
        train_loader = dataset["train"]
        eval_loader = dataset["test"]
        using_preprocessed = True
    else:
        print("Preprocessed dataset not found, loading from source...")
        dataset = load_func(cfg)
        # Split dataset for train/eval
        if isinstance(dataset, dict) or hasattr(dataset, "keys"):
            if "train" in dataset:
                dataset = dataset["train"]

        # Split dataset for train/eval
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_loader = dataset["train"]
        eval_loader = dataset["test"]

    if cfg.eval_subset_size > 0:
        print(f"Limiting validation set to {cfg.eval_subset_size} samples for speed")
        eval_loader = eval_loader.select(range(min(len(eval_loader), cfg.eval_subset_size)))
    # Use SFTConfig (compatible with TRL 0.26.2)
    training_args = SFTConfig(
        output_dir=model_dirname,
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
        optim=cfg.optim,
        run_name=model_output,
        save_strategy="steps",
        save_steps=cfg.eval_steps,  # Sync save with eval to capture best model immediately
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        dataset_text_field="text",
        logging_first_step=True,
        logging_nan_inf_filter=False,
    )

    # When using preprocessed data, chat template is already applied - don't apply it again!
    # When loading from source, SFTTrainer needs to format the data

    # CRITICAL: Configure response_template to train ONLY on assistant responses
    # This prevents the model from learning to predict system prompts or user inputs
    # For Granite models, the assistant response starts after the assistant role marker
    # We need to identify the exact token pattern from the chat template

    # Granite chat template format (typical):
    # <|start_of_role|>system<|end_of_role|>SYSTEM_TEXT<|start_of_role|>user<|end_of_role|>USER_TEXT<|start_of_role|>assistant<|end_of_role|>ASSISTANT_TEXT
    # We want to train only on ASSISTANT_TEXT

    # The response template should match the pattern that marks the start of assistant response
    # For Granite: "<|start_of_role|>assistant<|end_of_role|>"
    response_template = "<|start_of_role|>assistant<|end_of_role|>"

    from kurtis.collator import AssistantMaskingCollator

    # Use custom AssistantMaskingCollator to mask user/system prompts
    data_collator = AssistantMaskingCollator(
        tokenizer=tokenizer, response_template=response_template, mlm=False
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_loader,
        "eval_dataset": eval_loader,
        "args": training_args,
        "peft_config": lora_config,
        "data_collator": data_collator,
        "callbacks": [
            ValidationLossCallback(),
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    }

    if using_preprocessed:
        # Preprocessed data already has chat template applied in 'text' field
        # Tell SFTTrainer to NOT apply formatting
        print("Using preprocessed data - skipping chat template formatting in trainer")
        print(f"Using response_template for masking: {response_template}")
        trainer_kwargs["formatting_func"] = None
    else:
        # Raw data needs formatting - Apply formatting manually to avoid SFTTrainer conflict
        print("Using raw data - Applying chat template manually...")
        print(f"Using response_template for masking: {response_template}")

        # Apply formatting to train and eval datasets
        # formatting_prompts_func returns a list for a batch. We use batched mapping.
        column_names = train_loader.column_names

        def apply_format(examples):
            return {"text": formatting_prompts_func(examples)}

        trainer_kwargs["train_dataset"] = train_loader.map(
            apply_format, batched=True, remove_columns=column_names
        )
        trainer_kwargs["eval_dataset"] = eval_loader.map(
            apply_format, batched=True, remove_columns=column_names
        )

        # Now we don't need formatting_func in trainer
        trainer_kwargs["formatting_func"] = None

    trainer = SFTTrainer(**trainer_kwargs)

    # Fix for LoRA + SFTTrainer: Enable eval_loss printing
    # When using LoRA, the PeftModel wrapper can confuse Trainer's signature detection,
    # causing it to default can_return_loss to False. We explicitly enable it here.
    trainer.can_return_loss = True

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
    # Save final model (will use model's built-in chat template)
    save_and_merge_model(
        final_checkpoint_dir,
        final_output_merged_dir,
        hf_repo_id=hf_repo_id,
        push=push,
    )
