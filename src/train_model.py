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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
import yaml

try:
    from .analysis.failures import analyze_failures_from_test
    from .constants import RANDOM_SEED
    from .feature_engineering import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        prepare_training_data,
    )
    from .utils.metrics import compute_metrics_with_ci
    from .utils.plotting import (
        plot_calibration_curve,
        plot_feature_importance,
        plot_pr_with_ci,
        plot_roc_with_ci,
    )
    from .utils.repro import set_global_seed
except ImportError:  # pragma: no cover - fallback for script execution
    from analysis.failures import analyze_failures_from_test
    from constants import RANDOM_SEED
    from feature_engineering import (
        CATEGORICAL_FEATURES,
        NUMERIC_FEATURES,
        prepare_training_data,
    )
    from utils.metrics import compute_metrics_with_ci
    from utils.plotting import (
        plot_calibration_curve,
        plot_feature_importance,
        plot_pr_with_ci,
        plot_roc_with_ci,
    )
    from utils.repro import set_global_seed

try:
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from logging_utils import configure_logging, get_logger

# Define project structure and default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

logger = get_logger(__name__)


def _resolve_path(base_path: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_path / path).resolve()


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""

    config_data = yaml.safe_load(config_path.read_text()) or {}

    data_section = config_data.get("data", {})
    dataset_path_str = data_section.get("dataset_path", str(DATA_PATH))
    dataset_path = _resolve_path(config_path.parent, dataset_path_str)

    preprocessing_section = config_data.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        include_numeric_features=preprocessing_section.get("include_numeric_features"),
        exclude_numeric_features=preprocessing_section.get("exclude_numeric_features"),
        include_categorical_features=preprocessing_section.get("include_categorical_features"),
        exclude_categorical_features=preprocessing_section.get("exclude_categorical_features"),
    )

    models_section = config_data.get("models") or {}
    if not models_section:
        raise ValueError("Experiment config must define at least one model under `models`.")

    model_configs = {}
    for name, payload in models_section.items():
        payload = payload or {}
        model_configs[name] = ModelConfig(
            name=name,
            type=payload.get("type", name),
            hyperparameters=payload.get("hyperparameters", {}),
            model_parameters=payload.get("model_parameters", {}),
        )

    cv_section = config_data.get("cross_validation", {})
    cross_validation = CrossValidationConfig(
        n_splits=cv_section.get("n_splits", 5),
        shuffle=cv_section.get("shuffle", True),
        random_state=cv_section.get("random_state"),
    )

    return ExperimentConfig(
        dataset_path=dataset_path,
        preprocessing=preprocessing,
        models=model_configs,
        cross_validation=cross_validation,
    )


def _apply_feature_overrides(
    base_features: Sequence[str], include: Optional[Iterable[str]], exclude: Optional[Iterable[str]]
) -> List[str]:
    if include:
        filtered = [feature for feature in include if feature in base_features]
    else:
        filtered = list(base_features)

    if exclude:
        exclude_set = set(exclude)
        filtered = [feature for feature in filtered if feature not in exclude_set]

    return filtered


def resolve_feature_lists(preprocessing_config: Optional[PreprocessingConfig]) -> Tuple[List[str], List[str]]:
    """Resolve numeric and categorical feature lists based on config overrides."""

    if preprocessing_config is None:
        numeric = list(NUMERIC_FEATURES)
        categorical = list(CATEGORICAL_FEATURES)
    else:
        numeric = _apply_feature_overrides(
            NUMERIC_FEATURES,
            preprocessing_config.include_numeric_features,
            preprocessing_config.exclude_numeric_features,
        )
        categorical = _apply_feature_overrides(
            CATEGORICAL_FEATURES,
            preprocessing_config.include_categorical_features,
            preprocessing_config.exclude_categorical_features,
        )

    if not numeric and not categorical:
        raise ValueError("Preprocessing configuration removed all features.")

    return numeric, categorical


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


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a single estimator family."""

    name: str
    type: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for feature selection during preprocessing."""

    include_numeric_features: Optional[List[str]] = None
    exclude_numeric_features: Optional[List[str]] = None
    include_categorical_features: Optional[List[str]] = None
    exclude_categorical_features: Optional[List[str]] = None


@dataclass(frozen=True)
class CrossValidationConfig:
    """Configuration parameters for cross-validation."""

    n_splits: int = 5
    shuffle: bool = True
    random_state: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Represents the structure of a YAML-driven experiment."""

    dataset_path: Path
    preprocessing: PreprocessingConfig
    models: Dict[str, ModelConfig]
    cross_validation: CrossValidationConfig


DEFAULT_MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "logistic_regression": ModelConfig(
        name="logistic_regression",
        type="logistic_regression",
        hyperparameters={"model__C": [0.1, 1.0, 10.0]},
        model_parameters={"max_iter": 1000, "solver": "lbfgs"},
    ),
    "random_forest": ModelConfig(
        name="random_forest",
        type="random_forest",
        hyperparameters={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 10, 16],
            "model__min_samples_split": [2, 5],
        },
        model_parameters={"n_jobs": -1},
    ),
    "gradient_boosting": ModelConfig(
        name="gradient_boosting",
        type="gradient_boosting",
        hyperparameters={
            "model__n_estimators": [150, 250],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ),
}

DEFAULT_CROSS_VALIDATION_CONFIG = CrossValidationConfig(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def build_preprocessor(
    numeric_features: Optional[Sequence[str]] = None, categorical_features: Optional[Sequence[str]] = None
) -> ColumnTransformer:
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
    numeric = list(numeric_features) if numeric_features is not None else list(NUMERIC_FEATURES)
    categorical = list(categorical_features) if categorical_features is not None else list(CATEGORICAL_FEATURES)

    logger.debug(
        "Building preprocessing pipelines for numeric (%s) and categorical (%s) features",
        numeric,
        categorical,
    )
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

    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_transformer, numeric))
    if categorical:
        transformers.append(("categorical", categorical_transformer, categorical))

    if not transformers:
        raise ValueError("At least one feature must be provided to the preprocessor.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


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
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=random_state)

    # Second split: 50% of temporary becomes validation, 50% becomes test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state
    )

    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def _create_estimator(model_type: str, random_state: int, overrides: Dict[str, Any]) -> Any:
    """Instantiate an estimator for the requested model family."""

    overrides = overrides or {}

    if model_type == "logistic_regression":
        params = {"max_iter": 1000, "solver": "lbfgs", "random_state": random_state}
        params.update(overrides)
        return LogisticRegression(**params)
    if model_type == "random_forest":
        params = {"random_state": random_state, "n_jobs": -1}
        params.update(overrides)
        return RandomForestClassifier(**params)
    if model_type == "gradient_boosting":
        params = {"random_state": random_state}
        params.update(overrides)
        return GradientBoostingClassifier(**params)

    raise ValueError(f"Unsupported model family: {model_type}")


def build_model_pipelines(
    preprocessor: ColumnTransformer,
    *,
    model_configs: Optional[Dict[str, ModelConfig]] = None,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Tuple[ImbPipeline, Dict[str, Any]]]:
    """Define candidate models and their hyperparameter grids."""

    configs = model_configs or DEFAULT_MODEL_CONFIGS
    pipelines: Dict[str, Tuple[ImbPipeline, Dict[str, Any]]] = {}
    for name, config in configs.items():
        estimator = _create_estimator(config.type, random_state, config.model_parameters)
        pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("sampler", RandomOverSampler(random_state=random_state)),
                ("model", estimator),
            ]
        )
        pipelines[name] = (pipeline, dict(config.hyperparameters))

    return pipelines


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

    # Cast numpy scalar returns to builtin float to satisfy typing (Dict[str, float])
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def train_models(
    splits: SplitData,
    random_state: int = 42,
    *,
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
    model_configs: Optional[Dict[str, ModelConfig]] = None,
    cv_config: Optional[CrossValidationConfig] = None,
) -> List[ModelResult]:
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
    cv_settings = cv_config or DEFAULT_CROSS_VALIDATION_CONFIG

    preprocessor = build_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)
    pipelines = build_model_pipelines(preprocessor, model_configs=model_configs, random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_settings.n_splits, shuffle=cv_settings.shuffle, random_state=cv_settings.random_state)
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
    logger.info(
        "Selected best model: %s (Validation ROC-AUC: %.4f)", best_model.name, best_model.val_metrics["roc_auc"]
    )
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


def save_split_summary(splits: SplitData, output_dir: Path) -> None:
    """Save split statistics to JSON and CSV.

    Parameters
    ----------
    splits : SplitData
        The data splits
    output_dir : Path
        Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute statistics
    summary = {
        "train": {
            "count": len(splits.y_train),
            "positive_count": int(splits.y_train.sum()),
            "positive_rate": float(splits.y_train.mean()),
        },
        "validation": {
            "count": len(splits.y_val),
            "positive_count": int(splits.y_val.sum()),
            "positive_rate": float(splits.y_val.mean()),
        },
        "test": {
            "count": len(splits.y_test),
            "positive_count": int(splits.y_test.sum()),
            "positive_rate": float(splits.y_test.mean()),
        },
    }

    # Save JSON
    json_path = output_dir / "split_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved split summary JSON to {json_path}")

    # Save CSV
    csv_records = []
    for split_name, split_stats in summary.items():
        csv_records.append(
            {
                "split": split_name,
                "count": split_stats["count"],
                "positive_count": split_stats["positive_count"],
                "positive_rate": split_stats["positive_rate"],
            }
        )
    df = pd.DataFrame(csv_records)
    csv_path = output_dir / "splits.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved split summary CSV to {csv_path}")


def evaluate_baselines(splits: SplitData) -> Dict[str, Dict[str, float]]:
    """Evaluate simple baseline models.

    Parameters
    ----------
    splits : SplitData
        The data splits

    Returns
    -------
    Dict[str, Dict[str, float]]
        Metrics for each baseline
    """
    import numpy as np
    from sklearn.dummy import DummyClassifier

    baselines = {}

    # Majority class baseline
    majority_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
    majority_clf.fit(splits.X_train, splits.y_train)
    y_pred = majority_clf.predict(splits.X_test)
    y_prob = majority_clf.predict_proba(splits.X_test)[:, 1]

    baselines["majority_class"] = {
        "accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "precision": float(precision_score(splits.y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(splits.y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(splits.y_test, y_prob)) if len(np.unique(y_prob)) > 1 else 0.5,
    }

    # Stratified random baseline
    stratified_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
    stratified_clf.fit(splits.X_train, splits.y_train)
    y_pred = stratified_clf.predict(splits.X_test)
    y_prob = stratified_clf.predict_proba(splits.X_test)[:, 1]

    baselines["stratified_random"] = {
        "accuracy": float(accuracy_score(splits.y_test, y_pred)),
        "precision": float(precision_score(splits.y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(splits.y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(splits.y_test, y_prob)),
    }

    logger.info("Evaluated baseline models")
    return baselines


def compute_and_save_metrics_with_ci(
    model: ImbPipeline, splits: SplitData, output_dir: Path, n_bootstrap: int = 1000
) -> Dict[str, Dict[str, float]]:
    """Compute test metrics with bootstrap confidence intervals.

    Parameters
    ----------
    model : ImbPipeline
        Trained model
    splits : SplitData
        Data splits
    output_dir : Path
        Output directory
    n_bootstrap : int
        Number of bootstrap resamples

    Returns
    -------
    Dict[str, Dict[str, float]]
        Metrics with CIs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    y_pred = model.predict(splits.X_test)
    y_prob = model.predict_proba(splits.X_test)[:, 1]

    # Compute metrics with CIs
    metrics_ci = compute_metrics_with_ci(
        y_true=splits.y_test.values,
        y_pred=y_pred,
        y_prob=y_prob,
        n_bootstrap=n_bootstrap,
        random_state=RANDOM_SEED,
    )

    # Save JSON
    json_path = output_dir / "metrics_with_ci.json"
    with open(json_path, "w") as f:
        json.dump(metrics_ci, f, indent=2)
    logger.info(f"Saved metrics with CI to {json_path}")

    # Save Markdown
    md_path = output_dir / "metrics_with_ci.md"
    with open(md_path, "w") as f:
        f.write("# Test Metrics with 95% Confidence Intervals\n\n")
        f.write("| Metric | Mean | 95% CI Lower | 95% CI Upper |\n")
        f.write("|--------|------|--------------|---------------|\n")
        for metric_name, values in metrics_ci.items():
            f.write(f"| {metric_name} | {values['mean']:.3f} | {values['ci_lower']:.3f} | {values['ci_upper']:.3f} |\n")
    logger.info(f"Saved metrics markdown to {md_path}")

    return metrics_ci


def _get_feature_names_from_preprocessor(preprocessor: ColumnTransformer) -> List[str]:
    """Extract transformed feature names from a fitted preprocessor."""

    feature_names: List[str] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "numeric":
            feature_names.extend(list(columns))
        elif name == "categorical":
            columns_list = list(columns)
            if not columns_list:
                continue
            ohe = transformer.named_steps.get("onehot") if hasattr(transformer, "named_steps") else None
            if ohe is None:
                continue
            feature_names.extend(ohe.get_feature_names_out(columns_list))
    return feature_names


def generate_diagnostic_plots(model: ImbPipeline, splits: SplitData, output_dir: Path) -> None:
    """Generate all diagnostic plots.

    Parameters
    ----------
    model : ImbPipeline
        Trained model
    splits : SplitData
        Data splits
    output_dir : Path
        Output directory for figures
    """
    from sklearn.inspection import permutation_importance

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    y_prob = model.predict_proba(splits.X_test)[:, 1]

    # ROC curve
    logger.info("Generating ROC curve...")
    plot_roc_with_ci(y_true=splits.y_test.values, y_prob=y_prob, output_path=figures_dir / "roc.png", n_bootstrap=1000)

    # PR curve
    logger.info("Generating Precision-Recall curve...")
    plot_pr_with_ci(y_true=splits.y_test.values, y_prob=y_prob, output_path=figures_dir / "pr.png", n_bootstrap=1000)

    # Calibration curve
    logger.info("Generating calibration curve...")
    plot_calibration_curve(y_true=splits.y_test.values, y_prob=y_prob, output_path=figures_dir / "calibration.png")

    # Permutation importance
    logger.info("Computing permutation importance...")
    preprocessor = model.named_steps["preprocess"]
    X_test_transformed = preprocessor.transform(splits.X_test)

    # Get feature names based on the fitted preprocessor
    feature_names_out = _get_feature_names_from_preprocessor(preprocessor)

    perm_importance = permutation_importance(
        model.named_steps["model"], X_test_transformed, splits.y_test, n_repeats=10, random_state=RANDOM_SEED, n_jobs=-1
    )

    plot_feature_importance(
        feature_names=feature_names_out,
        importance_values=perm_importance.importances_mean,
        output_path=figures_dir / "perm_importance.png",
        top_k=20,
    )

    logger.info("Generated all diagnostic plots")


def run_training(
    data_path: Path = DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    random_state: int = RANDOM_SEED,
    experiment_config: Optional[ExperimentConfig] = None,
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
    # Set global random seed
    set_global_seed(random_state)
    logger.info(f"Set global random seed to {random_state}")

    dataset_path = experiment_config.dataset_path if experiment_config else data_path
    preprocessing_config = experiment_config.preprocessing if experiment_config else None
    numeric_features, categorical_features = resolve_feature_lists(preprocessing_config)
    model_configs = experiment_config.models if experiment_config else DEFAULT_MODEL_CONFIGS
    cv_config = experiment_config.cross_validation if experiment_config else DEFAULT_CROSS_VALIDATION_CONFIG

    df = pd.read_csv(dataset_path)
    logger.info("Loaded dataset with %d rows from %s", len(df), dataset_path)

    X, y = prepare_features(df)
    splits = stratified_splits(X, y, random_state=random_state)

    # Save split summary
    save_split_summary(splits, REPORTS_DIR)

    # Evaluate baselines
    logger.info("Evaluating baseline models...")
    baselines = evaluate_baselines(splits)

    # Train models
    results = train_models(
        splits,
        random_state=random_state,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        model_configs=model_configs,
        cv_config=cv_config,
    )
    best_model = select_best_model(results)

    # Save the best model pipeline and the full metrics report
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model.best_estimator, model_path)
    logger.info("Persisted best model (%s) to %s", best_model.name, model_path)
    save_metrics(results, metrics_path)

    # Compute metrics with confidence intervals
    logger.info("Computing metrics with bootstrap confidence intervals...")
    compute_and_save_metrics_with_ci(best_model.best_estimator, splits, REPORTS_DIR, n_bootstrap=1000)

    # Generate diagnostic plots
    logger.info("Generating diagnostic plots...")
    generate_diagnostic_plots(best_model.best_estimator, splits, REPORTS_DIR)

    # Failure analysis
    logger.info("Running failure analysis...")
    analyze_failures_from_test(splits.X_test, splits.y_test, best_model.best_estimator, REPORTS_DIR, top_n=10)

    # Add baselines to metrics report
    metrics_with_baselines = json.loads(metrics_path.read_text())
    metrics_with_baselines["baselines"] = baselines
    metrics_path.write_text(json.dumps(metrics_with_baselines, indent=2))

    logger.info("Training pipeline complete!")
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
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--config", type=Path, help="Path to an experiment YAML configuration file")
    return parser.parse_args()


def main() -> None:
    """Entry point for the script.

    Parses arguments, runs the training pipeline, and prints summary metrics
    for the best model to the console.
    """
    configure_logging()
    args = parse_args()
    experiment_config = load_experiment_config(args.config) if args.config else None
    best_model = run_training(
        data_path=args.data,
        model_path=args.model_output,
        metrics_path=args.metrics_output,
        random_state=args.random_state,
        experiment_config=experiment_config,
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
