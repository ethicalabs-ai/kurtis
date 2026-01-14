import pytest
from transformers import AutoConfig


@pytest.fixture
def mock_tokenizer(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.apply_chat_template.return_value = "Mocked chat template output"
    return tokenizer


@pytest.fixture
def mock_model(mocker):
    model = mocker.Mock()
    model.config = mocker.Mock(spec=AutoConfig)
    return model


@pytest.fixture
def sample_training_config():
    return {
        "dataset_name": "mock/dataset",
        "dataset_split": "train",
        "model_name": "mock/model",
    }
