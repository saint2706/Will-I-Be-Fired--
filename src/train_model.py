"""Train termination prediction models on HRDataset_v14.csv."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .feature_engineering import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        prepare_training_data,
    )
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        prepare_training_data,
    )

try:
    from .logging_utils import get_logger, configure_logging
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import get_logger, configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

logger = get_logger(__name__)


@dataclass
class SplitData:
    """Container for stratified train/validation/test splits."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class ModelResult:
    """Captures the outcome of training a single estimator."""

    name: str
    best_estimator: ImbPipeline
    cv_best_score: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical features."""

    logger.debug("Building preprocessing pipelines")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
            ("categorical", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ]
    )


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y after cleaning."""

    logger.info("Preparing features and target from raw dataset")
    return prepare_training_data(df)


def stratified_splits(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> SplitData:
    """Generate stratified train/validation/test splits."""

    logger.info("Creating stratified train/validation/test splits")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_state,
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def build_model_pipelines(preprocessor: ColumnTransformer) -> Dict[str, Tuple[ImbPipeline, Dict[str, Iterable]]]:
    """Define candidate models and their hyperparameter grids."""

    logistic = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            (
                "model",
                LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42),
            ),
        ]
    )
    logistic_grid = {"model__C": [0.1, 1.0, 10.0]}

    forest = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 16],
        "model__min_samples_split": [2, 5],
    }

    gradient_boost = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            (
                "model",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    )
    gradient_grid = {
        "model__n_estimators": [150, 250],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3],
    }

    return {
        "logistic_regression": (logistic, logistic_grid),
        "random_forest": (forest, forest_grid),
        "gradient_boosting": (gradient_boost, gradient_grid),
    }


def evaluate_model(model: ImbPipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Compute classification metrics for the supplied dataset."""

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def train_models(
    splits: SplitData,
    random_state: int = 42,
) -> List[ModelResult]:
    """Train and evaluate candidate models."""

    preprocessor = build_preprocessor()
    pipelines = build_model_pipelines(preprocessor)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results: List[ModelResult] = []

    for name, (pipeline, grid) in pipelines.items():
        logger.info("Training model %s", name)
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
        search.fit(splits.X_train, splits.y_train)
        logger.info("Completed grid search for %s with best score %.3f", name, search.best_score_)

        best_estimator = search.best_estimator_
        val_metrics = evaluate_model(best_estimator, splits.X_val, splits.y_val)
        test_metrics = evaluate_model(best_estimator, splits.X_test, splits.y_test)

        results.append(
            ModelResult(
                name=name,
                best_estimator=best_estimator,
                cv_best_score=search.best_score_,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
            )
        )

    return results


def select_best_model(results: List[ModelResult]) -> ModelResult:
    """Select the model with the highest validation ROC-AUC."""

    return max(results, key=lambda result: result.val_metrics["roc_auc"])


def save_metrics(results: List[ModelResult], metrics_path: Path) -> None:
    """Persist evaluation metrics for later inspection."""

    serialisable = {}
    for result in results:
        serialisable[result.name] = {
            "cv_best_score": result.cv_best_score,
            "validation": result.val_metrics,
            "test": result.test_metrics,
        }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("Saved metrics report to %s", metrics_path)


def run_training(
    data_path: Path = DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    random_state: int = 42,
) -> ModelResult:
    """Execute the full training workflow and return the best model."""

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset with %d rows", len(df))
    X, y = prepare_features(df)
    splits = stratified_splits(X, y, random_state=random_state)

    results = train_models(splits, random_state=random_state)
    best_model = select_best_model(results)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model.best_estimator, model_path)
    logger.info("Persisted best model (%s) to %s", best_model.name, model_path)
    save_metrics(results, metrics_path)

    return best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train termination prediction models")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the input CSV file")
    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Destination for the trained model pipeline",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Destination for the metrics report (JSON)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    best_model = run_training(
        data_path=args.data,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        random_state=args.random_state,
    )

    print("Best model:", best_model.name)
    print("Validation metrics:", json.dumps(best_model.val_metrics, indent=2))
    print("Test metrics:", json.dumps(best_model.test_metrics, indent=2))


if __name__ == "__main__":
    main()
