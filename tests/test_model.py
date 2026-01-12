"""Tests for kurtis.model module"""

import os
from unittest.mock import MagicMock, patch

import pytest

from kurtis.model import load_model_and_tokenizer, load_tokenizer_only


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.TRANSFORMERS_MODEL_PRETRAINED = "test-model"
    return config


def test_load_tokenizer_only(mock_config):
    """Test loading only the tokenizer"""
    with patch("kurtis.model.AutoTokenizer.from_pretrained") as mock_auto:
        mock_tokenizer = MagicMock()
        mock_auto.return_value = mock_tokenizer

        tokenizer = load_tokenizer_only(mock_config)

        assert tokenizer == mock_tokenizer
        mock_auto.assert_called_once_with("test-model")
        assert tokenizer.pad_token == tokenizer.eos_token


def test_load_tokenizer_only_override_name(mock_config):
    """Test loading tokenizer with an overridden name"""
    with patch("kurtis.model.AutoTokenizer.from_pretrained") as mock_auto:
        load_tokenizer_only(mock_config, model_name="other-model")
        mock_auto.assert_called_once_with("other-model")


def test_load_tokenizer_only_missing_config():
    """Test behavior when config is missing required attribute"""
    config = object()  # No attribute
    with pytest.raises(
        ValueError, match="Config must have TRANSFORMERS_MODEL_PRETRAINED attribute"
    ):
        load_tokenizer_only(config)


def test_load_model_and_tokenizer(mock_config):
    """Test loading both model and tokenizer"""
    with patch("kurtis.model.AutoTokenizer.from_pretrained") as mock_tok_auto:
        with patch("kurtis.model.AutoModelForCausalLM.from_pretrained") as mock_mod_auto:
            mock_tok = MagicMock()
            mock_mod = MagicMock()
            mock_tok_auto.return_value = mock_tok
            mock_mod_auto.return_value = mock_mod

            model, tokenizer = load_model_and_tokenizer(mock_config)

            assert model == mock_mod
            assert tokenizer == mock_tok
            mock_tok_auto.assert_called_once_with("test-model")
            mock_mod_auto.assert_called_once()
            assert mock_mod_auto.call_args[0][0] == "test-model"


def test_load_model_and_tokenizer_with_output(mock_config, tmp_path):
    """Test loading model from an existing output directory"""
    # Create a dummy output directory
    output_dir = tmp_path / "model_output"
    output_dir.mkdir()
    # checkpoint_dir = output_dir / "final_merged_checkpoint"
    # We don't need to create the checkpoint_dir, just mock os.path.exists

    with patch("kurtis.model.AutoTokenizer.from_pretrained") as mock_tok_auto:
        with patch("kurtis.model.AutoModelForCausalLM.from_pretrained") as mock_mod_auto:
            with patch("os.path.exists", return_value=True):
                # Ensure it picks the local path
                model, tokenizer = load_model_and_tokenizer(
                    mock_config, model_output=str(output_dir)
                )

                expected_path = os.path.join(str(output_dir), "final_merged_checkpoint")
                assert mock_mod_auto.call_args[0][0] == expected_path
                # Tokenizer still loads from PRETRAINED name
                mock_tok_auto.assert_called_with("test-model")
