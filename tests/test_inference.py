"""Tests for kurtis.inference module"""

from unittest.mock import MagicMock

import pytest
import torch

from kurtis.inference import batch_inference


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    # Mock apply_chat_template to return some dummy input IDs
    tokenizer.apply_chat_template.return_value = "formatted text"

    # Mock BatchEncoding-like behavior
    def mock_to(device=None):
        return {
            "input_ids": torch.tensor([[1, 2, 3]] * 2),  # Default to 2 for basic test
            "attention_mask": torch.tensor([[1, 1, 1]] * 2),
        }

    # Improved mock to handle dynamic batch size
    def mock_call(inputs, **kwargs):
        batch_size = len(inputs) if isinstance(inputs, list) else 1
        res = MagicMock()
        res.input_ids = torch.tensor([[1, 2, 3]] * batch_size)
        res.attention_mask = torch.tensor([[1, 1, 1]] * batch_size)
        res.__getitem__.side_effect = lambda k: {
            "input_ids": res.input_ids,
            "attention_mask": res.attention_mask,
        }[k]
        res.to.return_value = res
        return res

    tokenizer.side_effect = mock_call
    tokenizer.decode.return_value = "assistant response"
    tokenizer.eos_token_id = 50256
    return tokenizer


@pytest.fixture
def mock_model():
    model = MagicMock()
    # Mock model generation output
    # For a batch of N, generate returns [N, seq_len]
    model.generate.side_effect = lambda input_ids, **kwargs: torch.tensor(
        [[1, 2, 3, 4, 5]] * input_ids.shape[0]
    )
    return model


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.QA_INSTRUCTION = "Default instruction"
    return config


def test_batch_inference_basic(mock_model, mock_tokenizer, mock_config):
    """Test batch_inference with simple inputs"""
    prompts = ["Hello", "How are you?"]
    responses = batch_inference(
        mock_model,
        mock_tokenizer,
        mock_config,
        prompts,
        max_length=10,
    )

    assert len(responses) == 2
    assert responses[0] == "assistant response"
    assert mock_model.generate.called
    assert mock_tokenizer.apply_chat_template.called


def test_batch_inference_with_instruction(mock_model, mock_tokenizer, mock_config):
    """Test batch_inference with custom instruction"""
    prompts = ["Hello"]
    custom_instruction = "You are a helpful assistant."
    batch_inference(
        mock_model,
        mock_tokenizer,
        mock_config,
        prompts,
        instruction=custom_instruction,
    )

    # Verify apply_chat_template was called with the custom instruction
    # Note: it's called once per prompt
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert call_args[0]["role"] == "system"
    assert call_args[0]["content"] == custom_instruction


def test_batch_inference_uses_config_default(mock_model, mock_tokenizer, mock_config):
    """Test batch_inference uses instruction from config if not provided"""
    prompts = ["Hello"]
    batch_inference(
        mock_model,
        mock_tokenizer,
        mock_config,
        prompts,
    )

    # Verify apply_chat_template was called with the config default
    call_args = mock_tokenizer.apply_chat_template.call_args[0][0]
    assert call_args[0]["role"] == "system"
    assert call_args[0]["content"] == mock_config.QA_INSTRUCTION
