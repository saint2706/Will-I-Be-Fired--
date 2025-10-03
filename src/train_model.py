"""Train termination prediction models on the HRDataset_v14.csv dataset.

This script orchestrates the end-to-end model training process, including:
- Loading and preprocessing the raw data.
- Splitting the data into training, validation, and test sets.
- Defining and training multiple candidate classification models using a
  hyperparameter grid search.
- Evaluating the best-performing model on held-out data.
- Persisting the final model pipeline and evaluation metrics to disk.

The main execution flow is managed by the `run_training` function. The script
can be configured via command-line arguments to specify input and output paths.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import configure_logging, get_logger

# Define project structure and default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"

logger = get_logger(__name__)


@dataclass
class SplitData:
    """A container for stratified train, validation, and test data splits.

    Attributes
    ----------
    X_train:
        Training features.
    X_val:
        Validation features.
    X_test:
        Test features.
    y_train:
        Training target variable.
    y_val:
        Validation target variable.
    y_test:
        Test target variable.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass
class ModelResult:
    """Captures the outcome of training a single estimator.

    Attributes
    ----------
    name:
        The human-readable name of the model (e.g., "random_forest").
    best_estimator:
        The trained scikit-learn pipeline object after grid search.
    cv_best_score:
        The best cross-validation score (ROC-AUC) achieved.
    val_metrics:
        A dictionary of performance metrics on the validation set.
    test_metrics:
        A dictionary of performance metrics on the test set.
    """

    name: str
    best_estimator: ImbPipeline
    cv_best_score: float
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric and categorical features.

    This function defines the transformations to be applied to the raw data:
    - **Numeric features:** Median imputation followed by standard scaling.
    - **Categorical features:** Most-frequent imputation followed by one-hot
      encoding. Unknown categories encountered during prediction are ignored.

    Returns
    -------
    A scikit-learn `ColumnTransformer` object ready to be included in a model
    pipeline.
    """
    logger.debug("Building preprocessing pipelines for numeric and categorical features")
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # Combine transformers into a single preprocessor object
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
            ("categorical", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",  # Drop any columns not explicitly transformed
    )


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return the feature matrix X and target vector y after cleaning.

    This is a convenience wrapper around `prepare_training_data` from the
    feature engineering module.

    Parameters
    ----------
    df:
        The raw input DataFrame.

    Returns
    -------
    A tuple containing the features DataFrame and the target Series.
    """
    logger.info("Preparing features and target from raw dataset")
    return prepare_training_data(df)


def stratified_splits(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> SplitData:
    """Generate stratified train, validation, and test splits.

    The data is split into 70% training, 15% validation, and 15% test sets.
    Stratification is performed on the target variable `y` to ensure that the
    class distribution is preserved across all splits.

    Parameters
    ----------
    X:
        The feature matrix.
    y:
        The target vector.
    random_state:
        A seed for the random number generator to ensure reproducibility.

    Returns
    -------
    A `SplitData` object containing the data partitions.
    """
    logger.info("Creating stratified train/validation/test splits (70/15/15)")
    # First split: 70% train, 30% temporary (for val/test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=random_state
    )

    # Second split: 50% of temporary becomes validation, 50% becomes test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def build_model_pipelines(
    preprocessor: ColumnTransformer,
) -> Dict[str, Tuple[ImbPipeline, Dict[str, Any]]]:
    """Define candidate models and their hyperparameter grids.

    This function returns a dictionary where each key is a model name and the
    value is a tuple containing:
    1. An `imblearn` pipeline that combines preprocessing, over-sampling (to
       handle class imbalance), and the estimator.
    2. A dictionary defining the hyperparameter grid for `GridSearchCV`.

    Parameters
    ----------
    preprocessor:
        The `ColumnTransformer` to be included in each pipeline.

    Returns
    -------
    A dictionary of model names to (pipeline, grid) tuples.
    """
    # Pipeline for Logistic Regression
    logistic = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("model", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]
    )
    logistic_grid = {"model__C": [0.1, 1.0, 10.0]}

    # Pipeline for Random Forest
    forest = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )
    forest_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 16],
        "model__min_samples_split": [2, 5],
    }

    # Pipeline for Gradient Boosting
    gradient_boost = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=42)),
            ("model", GradientBoostingClassifier(random_state=42)),
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
    """Compute classification metrics for the supplied dataset.

    Calculates accuracy, precision, recall, and ROC-AUC.

    Parameters
    ----------
    model:
        The trained model pipeline.
    X:
        The features of the dataset to evaluate.
    y:
        The true labels of the dataset.

    Returns
    -------
    A dictionary mapping metric names to their float values.
    """
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities),
    }


def train_models(splits: SplitData, random_state: int = 42) -> List[ModelResult]:
    """Train and evaluate all candidate models defined in the script.

    This function iterates through the models from `build_model_pipelines`,
    performs a grid search with 5-fold cross-validation for each, and then
    evaluates the best resulting estimator on the validation and test sets.

    Parameters
    ----------
    splits:
        The `SplitData` object containing train, validation, and test sets.
    random_state:
        The random seed for `StratifiedKFold` to ensure reproducibility.

    Returns
    -------
    A list of `ModelResult` objects, one for each trained model.
    """
    preprocessor = build_preprocessor()
    pipelines = build_model_pipelines(preprocessor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results: List[ModelResult] = []

    for name, (pipeline, grid) in pipelines.items():
        logger.info("Training model: %s", name)
        # Perform grid search with cross-validation
        search = GridSearchCV(pipeline, param_grid=grid, scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(splits.X_train, splits.y_train)
        logger.info("Completed grid search for %s with best score %.3f", name, search.best_score_)

        # Evaluate the best estimator on validation and test data
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
    """Select the best model from a list of results.

    The selection criterion is the highest ROC-AUC score on the validation set.

    Parameters
    ----------
    results:
        A list of `ModelResult` objects.

    Returns
    -------
    The `ModelResult` object corresponding to the best-performing model.
    """
    best_model = max(results, key=lambda result: result.val_metrics["roc_auc"])
    logger.info("Selected best model: %s (Validation ROC-AUC: %.4f)", best_model.name, best_model.val_metrics["roc_auc"])
    return best_model


def save_metrics(results: List[ModelResult], metrics_path: Path) -> None:
    """Persist evaluation metrics for all models to a JSON file.

    Parameters
    ----------
    results:
        A list of `ModelResult` objects containing the data to save.
    metrics_path:
        The file path for the output JSON report.
    """
    serialisable_results = {
        result.name: {
            "cv_best_score": result.cv_best_score,
            "validation": result.val_metrics,
            "test": result.test_metrics,
        }
        for result in results
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(serialisable_results, indent=2))
    logger.info("Saved metrics report to %s", metrics_path)


def run_training(
    data_path: Path = DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    random_state: int = 42,
) -> ModelResult:
    """Execute the full training workflow and return the best model result.

    This function chains all the steps together: data loading, feature prep,
    splitting, model training, model selection, and artifact persistence.

    Parameters
    ----------
    data_path:
        Path to the input CSV dataset.
    model_path:
        Path to save the final trained model.
    metrics_path:
        Path to save the JSON metrics report.
    random_state:
        The global random seed for reproducibility.

    Returns
    -------
    The `ModelResult` for the best-performing model.
    """
    df = pd.read_csv(data_path)
    logger.info("Loaded dataset with %d rows from %s", len(df), data_path)

    X, y = prepare_features(df)
    splits = stratified_splits(X, y, random_state=random_state)
    results = train_models(splits, random_state=random_state)
    best_model = select_best_model(results)

    # Save the best model pipeline and the full metrics report
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model.best_estimator, model_path)
    logger.info("Persisted best model (%s) to %s", best_model.name, model_path)
    save_metrics(results, metrics_path)

    return best_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train termination prediction models")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the input CSV file")
    parser.add_argument(
        "--model-output", type=Path, default=DEFAULT_MODEL_PATH, help="Destination for the trained model pipeline"
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=DEFAULT_METRICS_PATH, help="Destination for the metrics report (JSON)"
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    """Entry point for the script.

    Parses arguments, runs the training pipeline, and prints summary metrics
    for the best model to the console.
    """
    configure_logging()
    args = parse_args()
    best_model = run_training(
        data_path=args.data,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        random_state=args.random_state,
    )

    # Print a summary of the final results
    print("\n--- Training Complete ---")
    print(f"Best model: {best_model.name}")
    print("\nValidation metrics:")
    print(json.dumps(best_model.val_metrics, indent=2))
    print("\nTest metrics:")
    print(json.dumps(best_model.test_metrics, indent=2))


if __name__ == "__main__":
    main()
