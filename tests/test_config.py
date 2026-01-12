"""Tests for kurtis.defaults and configuration loading"""

from kurtis.defaults import TrainingConfig


def test_training_config_from_dict_full():
    """Test TrainingConfig.from_dict with all fields present"""
    data = {
        "dataset_name": "test-ds",
        "lr": 1e-4,
        "batch_size": 16,
        "num_train_epochs": 5,
        "bf16": True,
    }
    cfg = TrainingConfig.from_dict(data)
    assert cfg.dataset_name == "test-ds"
    assert cfg.lr == 1e-4
    assert cfg.batch_size == 16
    assert cfg.num_train_epochs == 5
    assert cfg.bf16 is True


def test_training_config_from_dict_defaults():
    """Test TrainingConfig.from_dict with missing fields uses defaults"""
    data = {"dataset_name": "test-ds"}
    cfg = TrainingConfig.from_dict(data)
    assert cfg.dataset_name == "test-ds"
    assert cfg.lr == 3e-4  # Default value
    assert cfg.batch_size == 1  # Default value


def test_training_config_extra_fields_ignored():
    """Test that extra fields in the input dict are ignored"""
    data = {"dataset_name": "test-ds", "unknown_field": "some-value"}
    cfg = TrainingConfig.from_dict(data)
    assert cfg.dataset_name == "test-ds"
    assert not hasattr(cfg, "unknown_field")


def test_training_config_type_coercion():
    """Test that fields are present even if types are slightly off in input (if possible)"""
    # Dataclasses don't auto-coerce by default, but we should ensure it doesn't crash
    data = {"batch_size": "8", "num_train_epochs": 2.5}
    cfg = TrainingConfig.from_dict(data)
    # Note: In a real scenario, we might want to add validation/coercion in from_dict
    assert cfg.batch_size == "8"
    assert cfg.num_train_epochs == 2.5
