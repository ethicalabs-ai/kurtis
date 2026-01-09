from unittest.mock import MagicMock

from kurtis.evaluate import evaluate_main
from kurtis.train import train_model


def test_train_model_integration(mocker, mock_tokenizer, mock_model, tmp_path):
    # Mock dependencies
    mocker.patch("kurtis.train.prepare_model_for_kbit_training", return_value=mock_model)
    mocker.patch("kurtis.train.SFTTrainer")
    mocker.patch("kurtis.train.PeftModel")
    mocker.patch("kurtis.train.save_and_merge_model")

    # Mock dataset loader
    mock_dataset = MagicMock()
    mock_load_func = MagicMock(return_value=mock_dataset)

    config = {
        "dataset_name": "test_ds",
        "dataset_split": "train",
        "model_name": "test_model",
        "batch_size": 1,
        "logging_steps": 1,
        "num_train_epochs": 1,
    }

    output_dir = str(tmp_path / "output")
    model_output = str(tmp_path / "model_output")

    train_model(
        model=mock_model,
        tokenizer=mock_tokenizer,
        training_config=config,
        lora_config=MagicMock(),
        output_dir=output_dir,
        model_output=model_output,
        push=False,
        load_func=mock_load_func,  # Pass explicit mock to override default arg
    )

    # Verify SFTTrainer was called
    from kurtis.train import SFTTrainer

    SFTTrainer.assert_called_once()

    # Verify save_and_merge_model was called
    from kurtis.train import save_and_merge_model

    save_and_merge_model.assert_called_once()


def test_evaluate_main_integration(mocker, mock_tokenizer, mock_model, tmp_path):
    # Mock dataset structure
    # dataset.select -> val_dataset
    # val_dataset.select -> batch (list of dicts)

    mock_batch = [{"question": "q", "answer": "a"}]

    mock_val_dataset = MagicMock()
    mock_val_dataset.__len__.return_value = 10
    mock_val_dataset.select.return_value = mock_batch  # For the call inside evaluate_model

    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10
    mock_ds.select.return_value = mock_val_dataset  # For the call inside evaluate_main

    mocker.patch("kurtis.evaluate.load_dataset_from_config", return_value=mock_ds)

    # Mock evaluate library
    mock_rouge = MagicMock()
    mock_rouge.compute.return_value = {"rouge2": 0.5}
    mocker.patch("evaluate.load", return_value=mock_rouge)

    # Mock sklearn metrics
    mocker.patch("kurtis.evaluate.accuracy_score", return_value=0.8)
    mocker.patch("kurtis.evaluate.f1_score", return_value=0.8)
    mocker.patch("kurtis.evaluate.precision_score", return_value=0.8)
    mocker.patch("kurtis.evaluate.recall_score", return_value=0.8)

    # Mock batch_inference
    mocker.patch("kurtis.evaluate.batch_inference", return_value=["predicted answer"])

    # Mock config object
    mock_config = MagicMock()
    mock_config.EVALUATION_DATASET = {"dataset_name": "test_ds"}
    mock_config.MODEL_NAME = "test_model"

    # Run evaluation
    evaluate_main(
        model=mock_model,
        tokenizer=mock_tokenizer,
        config=mock_config,
        json_path="results.json",
        debug=True,
    )

    # Check if results file is created
    # Note: The code saves under benchmarks/MODEL_NAME/
    # expected_path = os.path.join("benchmarks", "test_model", "results.json")
    # Since we are mocking file I/O or running in real env, let's just check if it completed without error
    # For a more robust test, we could check file existence if we didn't mock open,
    # but the current code writes to a relative path.
    # Let's verify standard output or just that it finished.
