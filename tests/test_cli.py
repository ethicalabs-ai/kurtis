"""Tests for the kurtis CLI and subcommands"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from kurtis.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    """Test the main CLI help command"""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "[OPTIONS] COMMAND [ARGS]" in result.output
    assert "model" in result.output
    assert "dataset" in result.output


def test_dataset_preprocess_help(runner):
    """Test the dataset preprocess help command"""
    result = runner.invoke(cli, ["dataset", "preprocess", "--help"])
    assert result.exit_code == 0
    assert "--output-path" in result.output
    assert "--max-seq-length" in result.output


def test_model_train_help(runner):
    """Test the model train help command"""
    result = runner.invoke(cli, ["model", "train", "--help"])
    assert result.exit_code == 0
    assert "--output-dir" in result.output
    assert "--preprocessed-dataset-path" in result.output


def test_model_evaluate_help(runner):
    """Test the model evaluate help command"""
    result = runner.invoke(cli, ["model", "evaluate", "--help"])
    assert result.exit_code == 0
    assert "--output-dir" in result.output


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.TRANSFORMERS_MODEL_PRETRAINED = "test-model"
    cfg.MODEL_NAME = "test"
    cfg.TRAINING_CONFIG = {}
    cfg.LORA_CONFIG = MagicMock()
    cfg.HF_REPO_ID = "test/repo"
    cfg.QA_INSTRUCTION = "test instruction"
    return cfg


def test_model_train_invokes_handler(runner, mocker, mock_config):
    """Test that model train command invokes handle_train with correct args"""
    # Patch handle_train at the command level
    mock_handle = mocker.patch("kurtis.commands.model.train.handle_train")
    mocker.patch("torch.backends.mps.is_available", return_value=False)
    mocker.patch("os.makedirs")

    # Mock the context object used by Click
    with runner.isolated_filesystem():
        # Setup context manually if needed, but easier to patch where it's used
        with patch("kurtis.commands.model.train.click.pass_context", lambda f: f):
            # This is tricky because Click passes context via decorator.
            # Let's patch the entire command function or the logic inside.
            # Actually, the simplest way is to invoke it and patch the config loading.

            with patch("kurtis.cli.load_config", return_value=mock_config):
                result = runner.invoke(cli, ["model", "train", "--model-name", "overridden-model"])

    assert result.exit_code == 0
    # The config should have been updated
    assert mock_config.TRANSFORMERS_MODEL_PRETRAINED == "overridden-model"
    assert mock_handle.called


def test_dataset_preprocess_invokes_logic(runner, mocker, mock_config):
    """Test that dataset preprocess command invokes preprocessing_main with correct params"""
    mock_preprocess = mocker.patch("kurtis.commands.dataset.preprocess.preprocessing_main")

    with patch("kurtis.cli.load_config", return_value=mock_config):
        result = runner.invoke(
            cli,
            ["dataset", "preprocess", "--max-seq-length", "2048", "--output-path", "./custom_path"],
        )

    assert result.exit_code == 0
    assert mock_preprocess.called
    _, kwargs = mock_preprocess.call_args
    assert kwargs["max_length"] == 2048
    assert kwargs["output_path"] == "./custom_path"
