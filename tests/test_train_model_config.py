from pathlib import Path

import pandas as pd
import yaml

from src import train_model


def test_run_training_with_toy_config(tmp_path, monkeypatch):
    dataset_path = Path("HRDataset_v14.csv")
    df = pd.read_csv(dataset_path)
    positives = df[df["Termd"] == 1].head(12)
    negatives = df[df["Termd"] == 0].head(12)
    toy_df = pd.concat([positives, negatives]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    toy_csv = tmp_path / "toy_dataset.csv"
    toy_df.to_csv(toy_csv, index=False)

    config_payload = {
        "data": {"dataset_path": str(toy_csv)},
        "preprocessing": {
            "include_numeric_features": ["Salary", "EngagementSurvey"],
            "include_categorical_features": ["Department"],
        },
        "models": {
            "toy_logistic": {
                "type": "logistic_regression",
                "hyperparameters": {"model__C": [1.0]},
            }
        },
        "cross_validation": {"n_splits": 2, "shuffle": True, "random_state": 0},
    }
    config_path = tmp_path / "toy_config.yaml"
    config_path.write_text(yaml.safe_dump(config_payload))

    experiment_config = train_model.load_experiment_config(config_path)

    tmp_reports = tmp_path / "reports"
    monkeypatch.setattr(train_model, "REPORTS_DIR", tmp_reports)
    monkeypatch.setattr(train_model, "FIGURES_DIR", tmp_reports / "figures")
    monkeypatch.setattr(train_model, "compute_and_save_metrics_with_ci", lambda *args, **kwargs: {})
    monkeypatch.setattr(train_model, "generate_diagnostic_plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model, "analyze_failures_from_test", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_model, "save_split_summary", lambda *args, **kwargs: None)

    model_output = tmp_path / "model.joblib"
    metrics_output = tmp_path / "metrics.json"

    result = train_model.run_training(
        data_path=toy_csv,
        model_path=model_output,
        metrics_path=metrics_output,
        random_state=0,
        experiment_config=experiment_config,
    )

    assert result.name == "toy_logistic"
    assert model_output.exists()
    assert metrics_output.exists()
