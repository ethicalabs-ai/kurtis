import pytest
import yaml

from datasets import Dataset
from kurtis.dataset import filter_dataset_by_rule, load_dataset_from_config, load_datasets_from_yaml
from kurtis.defaults import TrainingConfig


@pytest.fixture
def sample_dataset():
    data = {
        "question": ["q1", "q2", "q3", "q4"],
        "answer": ["a1", "a2", "a3", "a4"],
        "tag": ["greeting", "tech", "greeting", "music"],
        "val": ["10", "20", "30", "40"],
    }
    return Dataset.from_dict(data)


def test_filter_dataset_by_rule_classes(sample_dataset):
    rule = {"classes": ["tag:greeting"]}
    filtered = filter_dataset_by_rule(sample_dataset, rule)
    assert len(filtered) == 2
    assert all(x["tag"] == "greeting" for x in filtered)


def test_filter_dataset_by_rule_max_samples(sample_dataset):
    # Rule must have classes to match anything currently
    rule = {"classes": ["tag:greeting"], "max_samples": 1}
    filtered = filter_dataset_by_rule(sample_dataset, rule)
    assert len(filtered) == 1


def test_filter_dataset_by_rule_combined(sample_dataset):
    rule = {"classes": ["tag:greeting"], "max_samples": 1}
    filtered = filter_dataset_by_rule(sample_dataset, rule)
    assert len(filtered) == 1
    assert filtered[0]["tag"] == "greeting"


def test_load_dataset_from_config_defaults(mocker):
    mock_load = mocker.patch("kurtis.dataset.load_dataset")
    mock_load.return_value = Dataset.from_dict({"question": ["q"], "answer": ["a"], "other": ["o"]})

    # If default defaults.py uses "prompt", we should supply it or change config
    cfg = TrainingConfig(dataset_name="dummy", prompt_column="question", response_column="answer")
    ds = load_dataset_from_config(cfg)

    # Check if standard columns are present and strings
    assert "question" in ds.column_names
    assert "answer" in ds.column_names
    assert ds[0]["question"] == "q"


def test_load_datasets_from_yaml(tmp_path, mocker):
    # Mock load_dataset to return a simple dataset
    mock_ds = Dataset.from_dict({"q_col": ["hi"], "a_col": ["byte"]})
    mocker.patch("kurtis.dataset.load_dataset", return_value=mock_ds)

    yaml_content = {
        "datasets": [
            {
                "path": "test/ds",
                "prompt_column": "q_col",
                "response_column": "a_col",
                "domain": "test_domain",
            }
        ]
    }

    config_file = tmp_path / "datasets.yaml"
    with open(config_file, "w") as f:
        yaml.dump(yaml_content, f)

    ds = load_datasets_from_yaml(str(config_file))

    assert len(ds) == 1
    assert ds[0]["question"] == "hi"
    assert ds[0]["answer"] == "byte"
    assert ds[0]["dataset_domain"] == "test_domain"
