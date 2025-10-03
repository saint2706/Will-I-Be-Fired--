"""Streamlit GUI for interactive termination risk exploration."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from .inference import (
        DEFAULT_MODEL_PATH,
        DEFAULT_TENURE_HORIZONS,
        TenureRisk,
        predict_tenure_risk,
    )
    from .logging_utils import configure_logging, get_logger
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
    from inference import (
        DEFAULT_MODEL_PATH,
        DEFAULT_TENURE_HORIZONS,
        TenureRisk,
        predict_tenure_risk,
    )
    from logging_utils import configure_logging, get_logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
DATE_FIELDS = ("DateofHire", "DOB", "LastPerformanceReview_Date")

configure_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Will I Be Fired?", layout="wide")
st.title("Employee Termination Risk Dashboard")
st.caption(
    "Estimate termination risk across tenure horizons, review model quality, "
    "and export reports for multiple employees."
)


@st.cache_data(show_spinner=False)
def load_reference_dataset() -> pd.DataFrame:
    """Load the training dataset for UI defaults."""

    logger.info("Loading reference dataset for GUI defaults from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def prepare_reference_metadata() -> tuple[dict[str, list], dict[str, float], dict[str, date]]:
    """Return dropdown options, numeric defaults, and date defaults."""

    dataset = load_reference_dataset()
    categorical_options: dict[str, list] = {}
    numeric_defaults: dict[str, float] = {}
    date_defaults: dict[str, date] = {}

    for column in CATEGORICAL_FEATURES:
        if column in dataset.columns:
            options = sorted(dataset[column].dropna().unique().tolist())
            categorical_options[column] = options or ["Unknown"]

    for column in NUMERIC_FEATURES:
        if column in dataset.columns:
            numeric_defaults[column] = float(dataset[column].dropna().median())

    for column in DATE_FIELDS:
        if column in dataset.columns:
            parsed = pd.to_datetime(dataset[column], errors="coerce")
            if parsed.notna().any():
                date_defaults[column] = parsed.dropna().median().date()

    today = pd.Timestamp.today().date()
    for column in DATE_FIELDS:
        date_defaults.setdefault(column, today)
    for column in NUMERIC_FEATURES:
        numeric_defaults.setdefault(column, 0.0)

    return categorical_options, numeric_defaults, date_defaults


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    """Load evaluation metrics from the training run."""

    if not METRICS_PATH.exists():
        logger.warning("Metrics file not found at %s", METRICS_PATH)
        return None
    logger.info("Loading metrics from %s", METRICS_PATH)
    return json.loads(METRICS_PATH.read_text())


def build_metrics_table(metrics: dict | None) -> pd.DataFrame | None:
    if not metrics:
        return None
    records = []
    for model_name, payload in metrics.items():
        for split in ("validation", "test"):
            for metric_name, value in payload[split].items():
                records.append(
                    {
                        "model": model_name,
                        "split": split,
                        "metric": metric_name,
                        "value": value,
                    }
                )
    return pd.DataFrame(records)


def _best_model_name(metrics: dict | None) -> str | None:
    if not metrics:
        return None
    return max(metrics.items(), key=lambda item: item[1]["validation"].get("roc_auc", 0.0))[0]


def build_combined_figure(best_metrics: dict | None, risks: Sequence[TenureRisk] | None) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Model Test Metrics", "Predicted Tenure Risk"),
        specs=[[{"type": "bar"}, {"type": "scatter"}]],
    )

    if best_metrics:
        metrics_names = ["accuracy", "precision", "recall", "roc_auc"]
        values = [best_metrics.get(metric, 0.0) for metric in metrics_names]
        fig.add_trace(
            go.Bar(x=metrics_names, y=values, name="Model Metrics"),
            row=1,
            col=1,
        )
        fig.update_yaxes(range=[0, 1], row=1, col=1)

    if risks:
        horizons = [risk.tenure_years for risk in risks]
        probabilities = [risk.termination_probability for risk in risks]
        confidences = [risk.confidence for risk in risks]
        fig.add_trace(
            go.Scatter(x=horizons, y=probabilities, mode="lines+markers", name="Termination Probability"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=horizons,
                y=confidences,
                mode="lines+markers",
                name="Confidence",
                line=dict(dash="dash"),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text="Tenure (years)", row=1, col=2)
        fig.update_yaxes(title_text="Probability", row=1, col=2, range=[0, 1])

    fig.update_layout(showlegend=True, height=500)
    return fig


categorical_options, numeric_defaults, date_defaults = prepare_reference_metadata()
metrics_payload = load_metrics()
metrics_table = build_metrics_table(metrics_payload)
best_model = _best_model_name(metrics_payload)
best_model_metrics = (
    metrics_payload.get(best_model, {}).get("test", {}) if best_model and metrics_payload else None
)

with st.sidebar:
    st.header("Prediction Settings")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    horizons = st.multiselect(
        "Tenure horizons (years)",
        options=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
        default=list(DEFAULT_TENURE_HORIZONS),
    )
    horizons = tuple(sorted(set(horizons))) or DEFAULT_TENURE_HORIZONS

    st.markdown("---")
    if metrics_table is not None:
        st.subheader("Latest Metrics")
        st.dataframe(metrics_table.pivot_table(index=["model"], columns="metric", values="value", aggfunc="first"))
    else:
        st.info("Run the training script to populate evaluation metrics.")

single_tab, batch_tab = st.tabs(["Single Employee", "Batch Upload"])

with single_tab:
    st.subheader("Predict risk for a single employee")
    risks_state: Sequence[TenureRisk] | None = None
    with st.form("single_employee_form"):
        columns = st.columns(3)
        record: dict[str, object] = {}

        for idx, column in enumerate(CATEGORICAL_FEATURES):
            options = categorical_options.get(column, ["Unknown"])
            with columns[idx % 3]:
                record[column] = st.selectbox(column, options=options, index=0)

        numeric_columns = st.columns(3)
        for idx, column in enumerate(NUMERIC_FEATURES):
            default_value = numeric_defaults.get(column, 0.0)
            with numeric_columns[idx % 3]:
                record[column] = st.number_input(column, value=float(default_value))

        date_columns = st.columns(3)
        for idx, column in enumerate(DATE_FIELDS):
            default_date = date_defaults.get(column, pd.Timestamp.today().date())
            with date_columns[idx % 3]:
                chosen_date = st.date_input(column, value=default_date)
                record[column] = chosen_date.isoformat()

        submitted = st.form_submit_button("Predict termination risk")

    if submitted:
        try:
            logger.info("Running single-record prediction")
            risks_state = predict_tenure_risk(record, horizons=horizons, model_path=Path(model_path))
        except Exception as exc:  # pragma: no cover - surface in UI
            logger.exception("Prediction failed")
            st.error(f"Prediction failed: {exc}")
            risks_state = None

    if submitted and risks_state:
        risk_df = pd.DataFrame(
            {
                "tenure_years": [risk.tenure_years for risk in risks_state],
                "termination_probability": [risk.termination_probability for risk in risks_state],
                "confidence": [risk.confidence for risk in risks_state],
            }
        )
        st.success("Prediction complete")
        st.dataframe(risk_df.style.format({"termination_probability": "{:.2%}", "confidence": "{:.2%}"}))

        combined_figure = build_combined_figure(best_model_metrics, risks_state)
        st.plotly_chart(combined_figure, use_container_width=True)
    elif submitted:
        combined_figure = build_combined_figure(best_model_metrics, None)
        st.plotly_chart(combined_figure, use_container_width=True)

with batch_tab:
    st.subheader("Upload CSV or JSON for batch predictions")
    uploaded = st.file_uploader("Upload employee records", type=["csv", "json"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            if uploaded.type == "application/json" or uploaded.name.endswith(".json"):
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
                if isinstance(payload, dict):
                    records_df = pd.DataFrame([payload])
                else:
                    records_df = pd.DataFrame(payload)
            else:
                records_df = pd.read_csv(uploaded)
        except Exception as exc:  # pragma: no cover - surface in UI
            logger.exception("Failed to parse uploaded file")
            st.error(f"Could not parse uploaded file: {exc}")
            records_df = None

        if records_df is not None and not records_df.empty:
            logger.info("Running batch predictions for %d employees", len(records_df))
            batch_results: list[dict[str, object]] = []
            for idx, (_, row) in enumerate(records_df.iterrows(), start=1):
                try:
                    risks = predict_tenure_risk(row, horizons=horizons, model_path=Path(model_path))
                except Exception as exc:  # pragma: no cover - continue processing
                    logger.exception("Prediction failed for row %d", idx)
                    st.warning(f"Prediction failed for row {idx}: {exc}")
                    continue
                for risk in risks:
                    batch_results.append(
                        {
                            "record_index": idx,
                            "tenure_years": risk.tenure_years,
                            "termination_probability": risk.termination_probability,
                            "confidence": risk.confidence,
                        }
                    )
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                st.success(f"Generated predictions for {batch_df['record_index'].nunique()} employees")
                st.dataframe(
                    batch_df.style.format(
                        {
                            "termination_probability": "{:.2%}",
                            "confidence": "{:.2%}",
                        }
                    )
                )
                st.dataframe(
                    batch_df.pivot_table(
                        index="record_index",
                        columns="tenure_years",
                        values="termination_probability",
                    ).style.format("{:.2%}")
                )

                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                report_path = REPORTS_DIR / f"prediction_report_{timestamp}.csv"
                batch_df.to_csv(report_path, index=False)
                logger.info("Saved batch report to %s", report_path)

                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download tidy report",
                    data=csv_bytes,
                    file_name=report_path.name,
                    mime="text/csv",
                )

                combined_figure = build_combined_figure(best_model_metrics, None)
                st.plotly_chart(combined_figure, use_container_width=True)
            else:
                st.warning("No predictions were generated from the uploaded data.")
        elif records_df is not None:
            st.warning("Uploaded file did not contain any rows.")
