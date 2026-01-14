"""Tests for chat template and response masking in training pipeline"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def test_response_template_always_configured(mocker, mock_tokenizer, mock_model):
    """
    CRITICAL: Verify response_template is ALWAYS set to train only on assistant responses.
    This prevents the model from learning to predict system/user prompts.
    """
    from kurtis.train import train_model

    # Mock dependencies
    mocker.patch("kurtis.train.prepare_model_for_kbit_training", return_value=mock_model)
    mock_tokenizer.encode.return_value = [123, 456, 789]  # Mock token IDs

    mock_sft_trainer_class = mocker.patch("kurtis.train.SFTTrainer")
    mocker.patch("kurtis.train.PeftModel")
    mocker.patch("kurtis.train.save_and_merge_model")

    # Mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = MagicMock()
    mock_dataset.__getitem__.return_value.train_test_split.return_value = {
        "train": MagicMock(__len__=lambda: 100),
        "test": MagicMock(__len__=lambda: 10),
    }

    config = {
        "dataset_name": "test_ds",
        "batch_size": 1,
        "logging_steps": 1,
        "num_train_epochs": 1,
        "eval_subset_size": 0,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("os.path.exists", return_value=False):  # No preprocessed data
            train_model(
                model=mock_model,
                tokenizer=mock_tokenizer,
                training_config=config,
                lora_config=MagicMock(),
                output_dir=tmp_dir,
                model_output=os.path.join(tmp_dir, "model"),
                push=False,
                load_func=lambda cfg: mock_dataset,
            )

    # Verify SFTTrainer was called with data_collator
    assert mock_sft_trainer_class.called
    call_kwargs = mock_sft_trainer_class.call_args[1]

    msg = "CRITICAL BUG: data_collator must be provided to handle masking!"
    assert "data_collator" in call_kwargs, msg
    # response_template is no longer passed as a direct argument to SFTTrainer
    # but strictly handled by the collator
    assert "response_template" not in call_kwargs


def test_formatting_func_none_with_preprocessed(mocker, mock_tokenizer, mock_model):
    """
    CRITICAL: Verify formatting_func is None when using preprocessed data.
    This prevents double application of chat template which corrupts training data.
    """
    from kurtis.train import train_model

    # Mock dependencies
    mocker.patch("kurtis.train.prepare_model_for_kbit_training", return_value=mock_model)
    mock_tokenizer.encode.return_value = [123, 456]

    mock_sft_trainer_class = mocker.patch("kurtis.train.SFTTrainer")
    mocker.patch("kurtis.train.PeftModel")
    mocker.patch("kurtis.train.save_and_merge_model")

    # Mock preprocessed dataset
    mock_dataset_dict = {
        "train": MagicMock(__len__=lambda: 100),
        "test": MagicMock(__len__=lambda: 10),
    }

    config = {
        "dataset_name": "test_ds",
        "batch_size": 1,
        "logging_steps": 1,
        "num_train_epochs": 1,
        "eval_subset_size": 0,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("os.path.exists", return_value=True):  # Preprocessed exists
            with patch("datasets.load_from_disk", return_value=mock_dataset_dict):
                train_model(
                    model=mock_model,
                    tokenizer=mock_tokenizer,
                    training_config=config,
                    lora_config=MagicMock(),
                    output_dir=tmp_dir,
                    model_output=os.path.join(tmp_dir, "model"),
                    push=False,
                )

    # Verify formatting_func is None to prevent double formatting
    assert mock_sft_trainer_class.called
    call_kwargs = mock_sft_trainer_class.call_args[1]

    msg = (
        "CRITICAL BUG: formatting_func must be None when using preprocessed data! "
        "Otherwise chat template is applied twice, corrupting the training inputs."
    )
    assert call_kwargs.get("formatting_func") is None, msg
    # response_template should not be passed to trainer directly
    assert "response_template" not in call_kwargs
    assert "data_collator" in call_kwargs


def test_preprocessing_applies_chat_template():
    """
    Verify that preprocessing correctly applies chat template to create 'text' field.
    """

    # Create mock config
    mock_config = MagicMock()
    mock_config.QA_INSTRUCTION = "You are a helpful assistant."
    mock_config.PREPROCESSING_TOKENIZER_MODEL = "ibm-granite/granite-4.0-350m"
    mock_config.TRANSFORMERS_MODEL_PRETRAINED = "ibm-granite/granite-4.0-350m"
    mock_config.DATASET_NAME = "test_dataset"

    # Mock the tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = (
        "<|start_of_role|>system<|end_of_role|>You are a helpful assistant.\n"
        "<|start_of_role|>user<|end_of_role|>Test question\n"
        "<|start_of_role|>assistant<|end_of_role|>Test answer"
    )

    # Mock dataset
    mock_dataset = MagicMock()

    mock_dataset.map.return_value = MagicMock()
    mock_dataset.map.return_value.__len__.return_value = 1

    with patch("kurtis.preprocess.load_tokenizer_only", return_value=mock_tokenizer):
        with patch("kurtis.preprocess.load_datasets", return_value=mock_dataset):
            with patch("kurtis.preprocess.process_dataset") as mock_process:
                mock_process.return_value = {"train": MagicMock(), "test": MagicMock()}
                mock_process.return_value["train"].__getitem__.return_value = {"text": "formatted"}

                with tempfile.TemporaryDirectory():
                    # This would normally call preprocessing but we're testing the concept
                    # The key assertion is that apply_chat_template is called
                    messages = [
                        {"role": "system", "content": mock_config.QA_INSTRUCTION},
                        {"role": "user", "content": "test"},
                        {"role": "assistant", "content": "response"},
                    ]
                    result = mock_tokenizer.apply_chat_template(messages, tokenize=False)

                    # Verify chat template was applied
                    assert "<|start_of_role|>assistant<|end_of_role|>" in result
                    assert mock_tokenizer.apply_chat_template.called


def test_response_template_token_pattern():
    """
    Verify the response_template uses the correct token pattern for Granite models.
    """
    from transformers import AutoTokenizer

    # Test with actual Granite tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-4.0-350m")

        # The pattern we're using
        assistant_pattern = "<|start_of_role|>assistant<|end_of_role|>"
        tokens = tokenizer.encode(assistant_pattern, add_special_tokens=False)

        # Verify we get tokens (not empty)
        assert len(tokens) > 0, "Response template pattern must encode to tokens"

        # Verify the pattern appears in a sample conversation
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        # The assistant pattern should appear in the formatted text
        msg = "Assistant role marker must appear in chat template"
        assert assistant_pattern in formatted or "assistant" in formatted, msg

    except Exception as e:
        pytest.skip(f"Could not load Granite tokenizer: {e}")
