"""Tests for kurtis.preprocess module"""

from unittest.mock import MagicMock, patch

import pytest

from kurtis.preprocess import prepare_initial_dataset, preprocessing_main


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.PREPROCESSING_TOKENIZER_MODEL = "tok-clean"
    config.TRANSFORMERS_MODEL_PRETRAINED = "tok-train"
    config.DATASET_NAME = "test-dataset"
    config.QA_INSTRUCTION = "Test system prompt"
    return config


def test_prepare_initial_dataset(mock_config):
    """Test the initial dataset preparation pipeline"""
    tokenizer = MagicMock()

    # Mock load_datasets to return a dummy dataset
    mock_ds = MagicMock()
    # Mock dataset.map and filter
    mock_ds.map.return_value = mock_ds
    mock_ds.filter.return_value = mock_ds
    mock_ds.remove_columns.return_value = mock_ds

    with patch("kurtis.preprocess.load_datasets", return_value=mock_ds):
        with patch("kurtis.preprocess.clean_and_truncate", side_effect=lambda x, m, t: x):
            result = prepare_initial_dataset(mock_config, tokenizer, 512)

            assert result == mock_ds
            assert mock_ds.map.called
            assert mock_ds.filter.called
            assert mock_ds.remove_columns.called


def test_preprocessing_main_local_save(mock_config):
    """Test the main preprocessing entry point with local saving"""
    # Mock dependencies
    with patch("kurtis.preprocess.load_tokenizer_only") as mock_load_tok:
        with patch("kurtis.preprocess.prepare_initial_dataset") as mock_prepare:
            with patch("kurtis.preprocess.process_dataset") as mock_process:
                with patch("nltk.download"):
                    with patch("torch.cuda.is_available", return_value=True):
                        # Setup mocks
                        mock_tok = MagicMock()
                        mock_load_tok.return_value = mock_tok

                        dummy_ds_split = MagicMock()
                        # dataset['train'] access
                        dummy_ds_split.__getitem__.side_effect = lambda k: {
                            "train": [{"text": "formatted text"}]
                        }[k]
                        dummy_ds_split.__len__.return_value = 10
                        dummy_ds_split.map.return_value = dummy_ds_split

                        mock_prepare.return_value = dummy_ds_split
                        mock_process.return_value = dummy_ds_split

                        preprocessing_main(mock_config, output_path="./test_out")

                        assert mock_load_tok.call_count == 2
                        assert mock_prepare.called
                        assert mock_process.called
                        assert dummy_ds_split.map.called
                        assert dummy_ds_split.save_to_disk.called
                        dummy_ds_split.save_to_disk.assert_called_with("./test_out")
