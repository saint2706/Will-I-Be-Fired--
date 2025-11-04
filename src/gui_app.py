"""Streamlit GUI for interactive termination risk exploration.

This script launches a web-based dashboard using Streamlit for comprehensive
analysis of employee termination risk. The application features:

- **Single Employee Prediction:** An interactive form to input details for a
  single employee and immediately see their predicted termination risk across
  different tenure horizons.
- **Batch Prediction:** A file uploader for processing multiple employee
  records from a CSV or JSON file, with results displayed in a table and
  made available for download.
- **Model Performance Visualization:** A chart displaying the key performance
  metrics (Accuracy, Precision, Recall, ROC-AUC) of the best-trained model
  on the test set.
- **Risk Trajectory Plotting:** A line chart that visualizes the predicted
  termination probability and the model's confidence over various tenure
  milestones.
- **Configuration:** A sidebar allows users to customize the model path and
  the tenure horizons for prediction.

Helper functions are used to load data, prepare UI components, and generate
visualizations, with heavy use of Streamlit's caching to ensure a responsive
user experience.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from .feature_engineering import CATEGORICAL_FEATURES, RAW_NUMERIC_INPUTS
    from .inference import (
        DEFAULT_MODEL_PATH,
        DEFAULT_TENURE_HORIZONS,
        TenureRisk,
        load_model,
        predict_tenure_risk,
    )
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import CATEGORICAL_FEATURES, RAW_NUMERIC_INPUTS
    from inference import (
        DEFAULT_MODEL_PATH,
        DEFAULT_TENURE_HORIZONS,
        TenureRisk,
        load_model,
        predict_tenure_risk,
    )
    from logging_utils import configure_logging, get_logger

# Define project structure and default paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATE_FIELDS = ("DateofHire", "DOB", "LastPerformanceReview_Date")

# --- Initialisation ---
configure_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Will I Be Fired?", layout="wide")
st.title("Employee Termination Risk Dashboard")
st.caption("Estimate risk, review model quality, and export reports.")


# --- Data Loading and Caching ---
@st.cache_data(show_spinner="Loading reference data...")
def load_reference_dataset() -> pd.DataFrame:
    """Load the source training dataset to populate UI defaults.

    This uses Streamlit's caching to avoid reloading the data on every
    interaction.

    Returns
    -------
    A pandas DataFrame with the source HR data.
    """
    logger.info("Loading reference dataset for GUI defaults from %s", DATA_PATH)
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Preparing UI metadata...")
def prepare_reference_metadata() -> Tuple[Dict[str, List], Dict[str, float], Dict[str, date]]:
    """Return dropdown options, numeric defaults, and date defaults for the UI.

    This function processes the reference dataset to extract sensible defaults
    for the input widgets, such as unique values for dropdowns and medians for
    numeric fields.

    Returns
    -------
    A tuple containing:
        - A dictionary for categorical feature options.
        - A dictionary for numeric feature default values.
        - A dictionary for date feature default values.
    """
    dataset = load_reference_dataset()
    categorical_options: Dict[str, List] = {}
    numeric_defaults: Dict[str, float] = {}
    date_defaults: Dict[str, date] = {}

    # Extract unique options for categorical dropdowns
    for column in CATEGORICAL_FEATURES:
        if column in dataset.columns:
            options = sorted(dataset[column].dropna().unique().tolist())
            categorical_options[column] = options or ["Unknown"]

    # Use median for numeric defaults
    for column in RAW_NUMERIC_INPUTS:
        if column in dataset.columns:
            numeric_defaults[column] = float(dataset[column].dropna().median())

    # Use median for date defaults
    for column in DATE_FIELDS:
        if column in dataset.columns:
            parsed = pd.to_datetime(dataset[column], errors="coerce")
            if parsed.notna().any():
                date_defaults[column] = parsed.dropna().median().date()

    # Fallback defaults for any missing columns
    today = pd.Timestamp.today().date()
    for column in DATE_FIELDS:
        date_defaults.setdefault(column, today)
    for column in RAW_NUMERIC_INPUTS:
        numeric_defaults.setdefault(column, 0.0)

    return categorical_options, numeric_defaults, date_defaults


@st.cache_data(show_spinner="Loading model metrics...")
def load_metrics() -> Optional[Dict]:
    """Load evaluation metrics from the `reports/metrics.json` file.

    Returns
    -------
    A dictionary containing the metrics, or None if the file doesn't exist.
    """
    if not METRICS_PATH.exists():
        logger.warning("Metrics file not found at %s", METRICS_PATH)
        return None
    logger.info("Loading metrics from %s", METRICS_PATH)
    return json.loads(METRICS_PATH.read_text())


@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_model(model_path: str):
    """Return a cached instance of the prediction pipeline for the given path."""

    return load_model(Path(model_path))


# --- UI and Plotting Helpers ---
def _is_model_entry(payload: Any) -> bool:
    """Check if a metrics payload represents a model entry (not baselines).

    Parameters
    ----------
    payload:
        A value from the metrics dictionary.

    Returns
    -------
    bool
        True if the payload has the expected model structure with validation/test splits.
    """
    if not isinstance(payload, dict):
        return False
    # Check that both validation and test keys exist and are dictionaries
    return (
        "validation" in payload
        and "test" in payload
        and isinstance(payload.get("validation"), dict)
        and isinstance(payload.get("test"), dict)
    )


def build_metrics_table(metrics: Optional[Dict]) -> Optional[pd.DataFrame]:
    """Transform the nested metrics dictionary into a flat DataFrame.

    Parameters
    ----------
    metrics:
        The raw dictionary loaded from `metrics.json`.

    Returns
    -------
    A tidy DataFrame with columns [model, split, metric, value], or None.
    """
    if not metrics:
        return None

    records = []
    for model_name, payload in metrics.items():
        if not _is_model_entry(payload):
            continue
        for split in ("validation", "test"):
            # Defensive check: ensure the split key exists and is a dict
            if split not in payload or not isinstance(payload[split], dict):
                logger.warning(
                    "Model '%s' is missing or has invalid '%s' metrics. Skipping this split.",
                    model_name,
                    split,
                )
                continue
            for metric_name, value in payload[split].items():
                records.append({"model": model_name, "split": split, "metric": metric_name, "value": value})
    
    return pd.DataFrame(records)


def _best_model_name(metrics: Optional[Dict]) -> Optional[str]:
    """Identify the best model based on the validation ROC-AUC score."""
    if not metrics:
        return None
    # Filter to only entries that have validation metrics (exclude baselines)
    model_entries = {name: payload for name, payload in metrics.items() if _is_model_entry(payload)}
    if not model_entries:
        return None
    return max(model_entries.items(), key=lambda item: item[1]["validation"].get("roc_auc", 0.0))[0]


def build_combined_figure(best_metrics: Optional[Dict], risks: Optional[Sequence[TenureRisk]]) -> go.Figure:
    """Create a combined Plotly figure with model metrics and risk predictions.

    The figure contains two subplots:
    1. A bar chart of the best model's performance on the test set.
    2. A scatter plot of predicted termination probability and confidence
       across different tenure horizons.

    Parameters
    ----------
    best_metrics:
        A dictionary of test metrics for the best model.
    risks:
        A sequence of `TenureRisk` objects from a prediction.

    Returns
    -------
    A Plotly `Figure` object.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Model Test Metrics", "Predicted Tenure Risk"),
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
    )

    # Subplot 1: Bar chart of model performance metrics
    if best_metrics:
        metric_names = ["accuracy", "precision", "recall", "roc_auc"]
        values = [best_metrics.get(metric, 0.0) for metric in metric_names]
        fig.add_trace(go.Bar(x=metric_names, y=values, name="Test Metrics"), row=1, col=1)
        fig.update_yaxes(title_text="Score", range=[0, 1], row=1, col=1)

    # Subplot 2: Line chart of tenure risk predictions
    if risks:
        horizons = [r.tenure_years for r in risks]
        probabilities = [r.termination_probability for r in risks]
        confidences = [r.confidence for r in risks]
        fig.add_trace(
            go.Scatter(x=horizons, y=probabilities, mode="lines+markers", name="Termination Probability"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=horizons, y=confidences, mode="lines+markers", name="Confidence", line_dash="dash"),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Tenure (years)", row=1, col=2)
        fig.update_yaxes(title_text="Probability", range=[0, 1], row=1, col=2)

    fig.update_layout(showlegend=True, height=500, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


# --- Main Application Logic ---
# Load all necessary metadata and metrics at the start.
categorical_options, numeric_defaults, date_defaults = prepare_reference_metadata()
metrics_payload = load_metrics()
metrics_table = build_metrics_table(metrics_payload)
best_model_name = _best_model_name(metrics_payload)
best_model_metrics = (
    metrics_payload.get(best_model_name, {}).get("test") if best_model_name and metrics_payload else None
)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    model_path_input = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    tenure_horizons = st.multiselect(
        "Tenure horizons (years)",
        options=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
        default=list(DEFAULT_TENURE_HORIZONS),
    )
    tenure_horizons = tuple(sorted(set(tenure_horizons))) or DEFAULT_TENURE_HORIZONS

    st.markdown("---")
    if metrics_table is not None:
        st.header("📊 Model Metrics")
        st.dataframe(
            metrics_table.pivot_table(index="model", columns="metric", values="value", aggfunc="first").style.format(
                "{:.3f}"
            )
        )
    else:
        st.info("Run `src/train_model.py` to generate evaluation metrics.")

# --- Main Panel (Tabs) ---
single_tab, batch_tab = st.tabs(["👤 Single Employee", "📁 Batch Upload"])

# --- Single Employee Prediction Tab ---
with single_tab:
    st.subheader("Predict risk for one employee")
    with st.form("single_employee_form"):
        form_cols = st.columns(3)
        record: Dict[str, Any] = {}

        # Dynamically create input widgets for all required features
        for i, col_name in enumerate(CATEGORICAL_FEATURES):
            with form_cols[i % 3]:
                record[col_name] = st.selectbox(col_name, options=categorical_options.get(col_name, ["Unknown"]))
        for i, col_name in enumerate(RAW_NUMERIC_INPUTS):
            with form_cols[i % 3]:
                record[col_name] = st.number_input(col_name, value=numeric_defaults.get(col_name, 0.0))
        for i, col_name in enumerate(DATE_FIELDS):
            with form_cols[i % 3]:
                chosen_date = st.date_input(col_name, value=date_defaults.get(col_name))
                record[col_name] = chosen_date.isoformat() if chosen_date else None

        submitted = st.form_submit_button("🚀 Predict Termination Risk")

    if submitted:
        try:
            logger.info("Running single-record prediction with horizons: %s", tenure_horizons)
            estimator = load_prediction_model(model_path_input)
            risks_state = predict_tenure_risk(record, horizons=tenure_horizons, model=estimator)
            st.success("Prediction complete!")
            # Display results in a table
            risk_df = pd.DataFrame([r.__dict__ for r in risks_state])
            st.dataframe(risk_df.style.format({"termination_probability": "{:.2%}", "confidence": "{:.2%}"}))
            # Display the combined visualization
            fig = build_combined_figure(best_model_metrics, risks_state)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.exception("Single-record prediction failed")
            st.error(f"Prediction failed: {e}")
    else:
        # Show the metrics chart even before the first prediction
        fig = build_combined_figure(best_model_metrics, None)
        st.plotly_chart(fig, use_container_width=True)


# --- Batch Upload Tab ---
with batch_tab:
    st.subheader("Upload a file for batch predictions")
    uploaded_file = st.file_uploader("Upload employee records", type=["csv", "json"], accept_multiple_files=False)

    if uploaded_file:
        try:
            # Parse uploaded file into a DataFrame
            if uploaded_file.name.endswith(".json"):
                payload = json.load(uploaded_file)
                records_df = pd.DataFrame(payload if isinstance(payload, list) else [payload])
            else:
                records_df = pd.read_csv(uploaded_file)
        except Exception as e:
            logger.exception("Failed to parse uploaded file")
            st.error(f"Could not parse file: {e}")
            records_df = None

        if records_df is not None and not records_df.empty:
            try:
                estimator = load_prediction_model(model_path_input)
            except Exception as e:
                logger.exception("Failed to load model for batch predictions")
                st.error(f"Could not load model: {e}")
            else:
                logger.info("Running batch predictions for %d employees", len(records_df))
                batch_results: List[Dict[str, Any]] = []
                progress_bar = st.progress(0)

                # Process each row and collect predictions
                for i, (_, row) in enumerate(records_df.iterrows()):
                    try:
                        risks = predict_tenure_risk(row.to_dict(), horizons=tenure_horizons, model=estimator)
                        for r in risks:
                            batch_results.append({"record_index": i + 1, **r.__dict__})
                    except Exception as e:
                        logger.warning("Prediction failed for row %d: %s", i + 1, e)
                        st.warning(f"Skipping prediction for row {i + 1} due to an error: {e}")
                    progress_bar.progress((i + 1) / len(records_df))

                if batch_results:
                    batch_df = pd.DataFrame(batch_results)
                    st.success(f"Generated predictions for {batch_df['record_index'].nunique()} employees.")

                    # Display results in a tidy format and as a pivot table
                    st.dataframe(batch_df.style.format({"termination_probability": "{:.2%}", "confidence": "{:.2%}"}))
                    st.subheader("Risk Probability Pivot Table")
                    pivot_table = batch_df.pivot_table(
                        index="record_index", columns="tenure_years", values="termination_probability"
                    )
                    st.dataframe(pivot_table.style.format("{:.2%}"))

                    # Provide a download button for the results
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    report_path = REPORTS_DIR / f"prediction_report_{timestamp}.csv"
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Full Report (CSV)",
                        data=csv_bytes,
                        file_name=report_path.name,
                        mime="text/csv",
                    )
                else:
                    st.warning("No valid predictions could be generated from the uploaded data.")
        elif records_df is not None:
            st.warning("The uploaded file is empty.")
