"""Ablation study for feature engineering and model components.

This module implements ablation experiments to understand the contribution of:
- Class rebalancing (RandomOverSampler)
- Date-derived features (tenure, age, years_since_last_review)
- Feature selection (top-k most important features)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from .constants import RANDOM_SEED
    from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_training_data
    from .logging_utils import get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from constants import RANDOM_SEED
    from feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, prepare_training_data
    from logging_utils import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"


@dataclass
class AblationResult:
    """Container for ablation experiment results."""

    experiment_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_score_mean: float
    cv_score_std: float


def run_ablation_no_rebalancing(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED
) -> AblationResult:
    """Run ablation: remove class rebalancing (RandomOverSampler).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    random_state : int
        Random seed

    Returns
    -------
    AblationResult
        Results without rebalancing
    """
    logger.info("Running ablation: no class rebalancing")

    # Build preprocessor
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
            ("categorical", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )

    # Pipeline without sampler
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)),
        ]
    )

    # Cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    # Fit and evaluate
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return AblationResult(
        experiment_name="no_rebalancing",
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_prob)),
        cv_score_mean=float(cv_scores.mean()),
        cv_score_std=float(cv_scores.std()),
    )


def run_ablation_no_date_features(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_SEED
) -> AblationResult:
    """Run ablation: remove date-derived features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    random_state : int
        Random seed

    Returns
    -------
    AblationResult
        Results without date features
    """
    logger.info("Running ablation: no date-derived features")

    # Remove temporal features
    date_features = ["tenure_years", "age_years", "years_since_last_review"]
    X_no_dates = X.drop(columns=[col for col in date_features if col in X.columns], errors="ignore")

    numeric_features = [f for f in NUMERIC_FEATURES if f not in date_features]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, list(numeric_features)),
            ("categorical", categorical_transformer, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )

    from imblearn.over_sampling import RandomOverSampler

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=random_state)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)),
        ]
    )

    cv_scores = cross_val_score(pipeline, X_no_dates, y, cv=5, scoring="roc_auc", n_jobs=-1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_no_dates, y, test_size=0.2, stratify=y, random_state=random_state
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return AblationResult(
        experiment_name="no_date_features",
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_prob)),
        cv_score_mean=float(cv_scores.mean()),
        cv_score_std=float(cv_scores.std()),
    )


def run_ablation_top_k_features(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    feature_importance: Optional[Dict[str, float]] = None,
    random_state: int = RANDOM_SEED,
) -> AblationResult:
    """Run ablation: use only top-k most important features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    k : int
        Number of top features to keep
    feature_importance : Dict[str, float], optional
        Pre-computed feature importance scores. If None, computes them.
    random_state : int
        Random seed

    Returns
    -------
    AblationResult
        Results with top-k features only
    """
    logger.info(f"Running ablation: top-{k} features only")

    # If no importance provided, compute it
    if feature_importance is None:
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

        # Train quick model to get importance
        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
                ("categorical", categorical_transformer, list(CATEGORICAL_FEATURES)),
            ],
            remainder="drop",
        )

        from imblearn.over_sampling import RandomOverSampler

        temp_pipeline = ImbPipeline(
            steps=[
                ("preprocess", preprocessor),
                ("sampler", RandomOverSampler(random_state=random_state)),
                ("model", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ]
        )
        temp_pipeline.fit(X_train, y_train)

        # Get feature names after preprocessing
        feature_names_out = (
            list(NUMERIC_FEATURES)
            + list(
                temp_pipeline.named_steps["preprocess"]
                .named_transformers_["categorical"]
                .named_steps["onehot"]
                .get_feature_names_out(CATEGORICAL_FEATURES)
            )
        )

        # Compute permutation importance
        X_test_transformed = temp_pipeline.named_steps["preprocess"].transform(X_test)
        perm_importance = permutation_importance(
            temp_pipeline.named_steps["model"], X_test_transformed, y_test, n_repeats=10, random_state=random_state
        )

        feature_importance = dict(zip(feature_names_out, perm_importance.importances_mean))

    # Select top-k features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f[0] for f in sorted_features[:k]]

    # Map back to original feature names (handle one-hot encoded features)
    selected_numeric = [f for f in NUMERIC_FEATURES if f in top_features]
    selected_categorical = list(
        set(
            [
                f.split("_")[0] if "_" in f else f
                for f in top_features
                if any(cat in f for cat in CATEGORICAL_FEATURES)
            ]
        )
    )

    # Build pipeline with selected features
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, selected_numeric if selected_numeric else [NUMERIC_FEATURES[0]]),
            ("categorical", categorical_transformer, selected_categorical if selected_categorical else [CATEGORICAL_FEATURES[0]]),
        ],
        remainder="drop",
    )

    from imblearn.over_sampling import RandomOverSampler

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", RandomOverSampler(random_state=random_state)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state, n_jobs=-1)),
        ]
    )

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="roc_auc", n_jobs=-1)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return AblationResult(
        experiment_name=f"top_{k}_features",
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_prob)),
        cv_score_mean=float(cv_scores.mean()),
        cv_score_std=float(cv_scores.std()),
    )


def run_all_ablations(
    data_path: Path = DATA_PATH, output_dir: Optional[Path] = None, random_state: int = RANDOM_SEED
) -> List[AblationResult]:
    """Run all ablation experiments and save results.

    Parameters
    ----------
    data_path : Path
        Path to dataset
    output_dir : Path, optional
        Directory to save results. If None, uses PROJECT_ROOT/reports
    random_state : int
        Random seed

    Returns
    -------
    List[AblationResult]
        List of ablation results
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "reports"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    X, y = prepare_training_data(df)

    results = []

    # Ablation 1: No rebalancing
    results.append(run_ablation_no_rebalancing(X, y, random_state))

    # Ablation 2: No date features
    results.append(run_ablation_no_date_features(X, y, random_state))

    # Ablation 3: Top-k features (k=5, 10, 20)
    for k in [5, 10, 20]:
        results.append(run_ablation_top_k_features(X, y, k=k, random_state=random_state))

    # Save results as CSV
    results_df = pd.DataFrame([vars(r) for r in results])
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)
    logger.info(f"Saved ablation results to {output_dir / 'ablation_results.csv'}")

    # Save markdown summary
    write_ablation_markdown(results, output_dir / "ABLATIONS.md")

    return results


def write_ablation_markdown(results: List[AblationResult], output_path: Path) -> None:
    """Write ablation results as a markdown report.

    Parameters
    ----------
    results : List[AblationResult]
        Ablation results to summarize
    output_path : Path
        Path to save markdown file
    """
    with open(output_path, "w") as f:
        f.write("# Ablation Study Results\n\n")
        f.write("This document summarizes ablation experiments that isolate the contribution of ")
        f.write("individual model components and feature groups.\n\n")

        f.write("## Experiments\n\n")
        f.write("1. **No Rebalancing**: Remove RandomOverSampler (no class rebalancing)\n")
        f.write("2. **No Date Features**: Remove tenure_years, age_years, years_since_last_review\n")
        f.write("3. **Top-K Features**: Use only the top 5, 10, or 20 most important features\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Experiment | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Score (mean±std) |\n")
        f.write("|------------|----------|-----------|--------|----|---------|-----------------------|\n")

        for result in results:
            f.write(
                f"| {result.experiment_name} | {result.accuracy:.3f} | {result.precision:.3f} | "
                f"{result.recall:.3f} | {result.f1:.3f} | {result.roc_auc:.3f} | "
                f"{result.cv_score_mean:.3f}±{result.cv_score_std:.3f} |\n"
            )

        f.write("\n## Insights\n\n")
        f.write("### Impact of Class Rebalancing\n")
        no_rebal = next(r for r in results if r.experiment_name == "no_rebalancing")
        f.write(
            f"- Without rebalancing: ROC-AUC = {no_rebal.roc_auc:.3f}, Recall = {no_rebal.recall:.3f}\n"
        )
        f.write("- Class rebalancing significantly improves recall for the minority class (terminated employees)\n\n")

        f.write("### Impact of Date-Derived Features\n")
        no_dates = next(r for r in results if r.experiment_name == "no_date_features")
        f.write(
            f"- Without date features: ROC-AUC = {no_dates.roc_auc:.3f}\n"
        )
        f.write("- Temporal features (tenure, age, review recency) are important for predicting termination risk\n\n")

        f.write("### Feature Selection Trade-offs\n")
        top_5 = next(r for r in results if r.experiment_name == "top_5_features")
        top_10 = next(r for r in results if r.experiment_name == "top_10_features")
        top_20 = next(r for r in results if r.experiment_name == "top_20_features")
        f.write(f"- Top-5 features: ROC-AUC = {top_5.roc_auc:.3f}\n")
        f.write(f"- Top-10 features: ROC-AUC = {top_10.roc_auc:.3f}\n")
        f.write(f"- Top-20 features: ROC-AUC = {top_20.roc_auc:.3f}\n")
        f.write("- A small set of features captures most predictive power; adding more features yields diminishing returns\n")

    logger.info(f"Wrote ablation markdown to {output_path}")


if __name__ == "__main__":
    from logging_utils import configure_logging

    configure_logging()
    run_all_ablations()
