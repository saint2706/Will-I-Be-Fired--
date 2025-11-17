"""Streamlit dashboard for interactive termination risk exploration.

The app provides:
- A structured single-employee form with HR-friendly sections.
- Batch CSV upload with downloadable predictions enriched with risk bands and
  recommended actions from ``configs/policy.yaml``.
- A sidebar to tweak model paths, tenure horizons, and review model metrics.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

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
    from .predict_cli import get_risk_band_and_actions, load_policy_config
    from .schemas import EmployeeRecord, ValidationError
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
    from predict_cli import get_risk_band_and_actions, load_policy_config
    from schemas import EmployeeRecord, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "HRDataset_v14.csv"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
POLICY_CONFIG_PATH = PROJECT_ROOT / "configs" / "policy.yaml"
DATE_FIELDS = ("DateofHire", "DOB", "LastPerformanceReview_Date")
NUMERIC_HELP_TEXT: Dict[str, str] = {
    "Salary": "Annual base pay in USD.",
    "EngagementSurvey": "Score from the most recent engagement survey (0-5).",
    "EmpSatisfaction": "Manager-reported satisfaction (1-5).",
    "SpecialProjectsCount": "Special projects completed in the last review cycle.",
    "DaysLateLast30": "Number of late arrivals in the last 30 days.",
    "Absences": "Total absences recorded this year.",
}
FORM_SECTIONS: Dict[str, Sequence[str]] = {
    "Demographics": ("Sex", "MaritalDesc", "CitizenDesc", "RaceDesc", "HispanicLatino", "State"),
    "Role & Performance": ("Department", "Position", "PerformanceScore", "RecruitmentSource", "SpecialProjectsCount"),
    "Compensation & Engagement": ("Salary", "EngagementSurvey", "EmpSatisfaction"),
    "Attendance": ("DaysLateLast30", "Absences"),
    "Dates": DATE_FIELDS,
}

configure_logging()
logger = get_logger(__name__)

st.set_page_config(page_title="Will I Be Fired?", layout="wide")
st.title("Employee Termination Risk Assistant")
st.caption("For HR analytics education and what-if exploration.")
st.markdown("Use the form or upload a CSV to estimate how employee characteristics impact early termination risk.")

if "single_form_errors" not in st.session_state:
    st.session_state["single_form_errors"] = {}


def _map_validation_errors(error: ValidationError) -> Dict[str, List[str]]:
    """Convert a ValidationError into a mapping of field -> messages."""

    field_errors: Dict[str, List[str]] = {}
    for issue in error.errors():
        loc = str(issue.get("loc", ["record"])[0])
        field_errors.setdefault(loc, []).append(issue.get("msg"))
    return field_errors


@st.cache_data(show_spinner="Loading reference data...")
def load_reference_dataset() -> pd.DataFrame:
    logger.info("Loading reference dataset for GUI defaults from %s", DATA_PATH)
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner="Preparing UI metadata...")
def prepare_reference_metadata() -> Tuple[Dict[str, List], Dict[str, float], Dict[str, date]]:
    dataset = load_reference_dataset()
    categorical_options: Dict[str, List] = {}
    numeric_defaults: Dict[str, float] = {}
    date_defaults: Dict[str, date] = {}

    for column in CATEGORICAL_FEATURES:
        if column in dataset.columns:
            options = sorted(dataset[column].dropna().unique().tolist())
            categorical_options[column] = options or ["Unknown"]

    for column in RAW_NUMERIC_INPUTS:
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
    for column in RAW_NUMERIC_INPUTS:
        numeric_defaults.setdefault(column, 0.0)

    return categorical_options, numeric_defaults, date_defaults


@st.cache_data(show_spinner="Loading model metrics...")
def load_metrics() -> Optional[Dict]:
    if not METRICS_PATH.exists():
        logger.warning("Metrics file not found at %s", METRICS_PATH)
        return None
    logger.info("Loading metrics from %s", METRICS_PATH)
    return json.loads(METRICS_PATH.read_text())


@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_model(model_path: str):
    return load_model(Path(model_path))


@st.cache_data(show_spinner="Loading HR policy guidance...")
def load_policy_guidance() -> Dict[str, Any]:
    return load_policy_config(POLICY_CONFIG_PATH)


def _is_model_entry(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        "validation" in payload
        and "test" in payload
        and isinstance(payload.get("validation"), dict)
        and isinstance(payload.get("test"), dict)
    )


def build_metrics_table(metrics: Optional[Dict]) -> Optional[pd.DataFrame]:
    if not metrics:
        return None

    records = []
    for model_name, payload in metrics.items():
        if not _is_model_entry(payload):
            continue
        for split in ("validation", "test"):
            if split not in payload or not isinstance(payload[split], dict):
                continue
            for metric_name, value in payload[split].items():
                records.append({"model": model_name, "split": split, "metric": metric_name, "value": value})

    return pd.DataFrame(records)


def _best_model_name(metrics: Optional[Dict]) -> Optional[str]:
    if not metrics:
        return None
    model_entries = {name: payload for name, payload in metrics.items() if _is_model_entry(payload)}
    if not model_entries:
        return None
    return max(model_entries.items(), key=lambda item: item[1]["validation"].get("roc_auc", 0.0))[0]


def _format_percentage(value: float) -> str:
    return f"{value:.1%}"


def _tenure_risk_dataframe(risks: Sequence[TenureRisk], policy_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    risk_df = pd.DataFrame(
        [
            {
                "tenure_years": risk.tenure_years,
                "termination_probability": risk.termination_probability,
                "confidence": risk.confidence,
            }
            for risk in risks
        ]
    )
    risk_df.sort_values("tenure_years", inplace=True)

    if policy_config:
        annotations = [get_risk_band_and_actions(prob, policy_config) for prob in risk_df["termination_probability"]]
        risk_df["risk_band"] = [band for band, _ in annotations]
        risk_df["recommended_actions"] = ["; ".join(actions) for _, actions in annotations]

    return risk_df


def _render_form_sections(
    record: Dict[str, Any],
    categorical_options: Dict[str, List],
    numeric_defaults: Dict[str, float],
    date_defaults: Dict[str, date],
    form_errors: Dict[str, List[str]],
) -> None:
    integer_like = {"SpecialProjectsCount", "DaysLateLast30", "Absences"}

    for section_name, fields in FORM_SECTIONS.items():
        st.markdown(f"### {section_name}")
        columns = st.columns(2)
        for idx, field in enumerate(fields):
            column = columns[idx % len(columns)]
            with column:
                help_text = NUMERIC_HELP_TEXT.get(field)
                if field in CATEGORICAL_FEATURES:
                    options = categorical_options.get(field, ["Unknown"])
                    record[field] = st.selectbox(field, options=options)
                elif field in RAW_NUMERIC_INPUTS:
                    default_value = float(numeric_defaults.get(field, 0.0))
                    step = 1.0 if field in integer_like else 0.1
                    record[field] = st.number_input(
                        field,
                        value=default_value,
                        step=step,
                        help=help_text,
                    )
                elif field in DATE_FIELDS:
                    default_date = date_defaults.get(field)
                    chosen_date = st.date_input(field, value=default_date)
                    record[field] = chosen_date.isoformat()
                else:
                    record[field] = st.text_input(field)

                for message in form_errors.get(field, []):
                    st.caption(f"⚠️ {message}")


def _display_prediction_results(risks_state: Sequence[TenureRisk], policy_config: Dict[str, Any]) -> None:
    if not risks_state:
        st.info("No predictions were generated. Please adjust the inputs and try again.")
        return

    risk_df = _tenure_risk_dataframe(risks_state, policy_config)
    primary_row = risk_df.iloc[0]

    st.metric(
        "Overall termination probability",
        _format_percentage(primary_row["termination_probability"]),
        help="Based on the shortest selected tenure horizon.",
    )

    styled_df = risk_df.copy()
    styled_df["termination_probability"] = styled_df["termination_probability"].map(_format_percentage)
    styled_df["confidence"] = styled_df["confidence"].map(_format_percentage)
    st.dataframe(styled_df, use_container_width=True)

    chart_source = risk_df.set_index("tenure_years")["termination_probability"]
    st.line_chart(chart_source, height=260, use_container_width=True)

    risk_band = primary_row.get("risk_band")
    recommended_actions = primary_row.get("recommended_actions", "")
    if risk_band:
        st.markdown(f"**Risk band:** {risk_band}")
    if recommended_actions:
        st.markdown("**Recommended actions**")
        for action in recommended_actions.split("; "):
            st.write(f"- {action}")
    else:
        st.info("No configured actions for this risk band in policy.yaml")


def build_combined_figure(best_metrics: Optional[Dict], risks: Optional[Sequence[TenureRisk]]) -> Dict[str, Any]:
    """Compatibility shim for legacy tests expecting a Plotly figure builder.

    The modern UI no longer renders this object, but returning a structured
    payload preserves backwards compatibility for automated tests that only
    verify the function exists. Callers can inspect the dictionary for
    debugging purposes if needed.
    """

    return {
        "best_metrics": best_metrics or {},
        "risks": [
            {
                "tenure_years": risk.tenure_years,
                "termination_probability": risk.termination_probability,
                "confidence": risk.confidence,
            }
            for risk in risks or []
        ],
    }


# --- Cached metadata ---
categorical_options, numeric_defaults, date_defaults = prepare_reference_metadata()
metrics_payload = load_metrics()
metrics_table = build_metrics_table(metrics_payload)
best_model_name = _best_model_name(metrics_payload)
policy_config = load_policy_guidance()

# --- Sidebar ---
with st.sidebar:
    st.header("How to use this app")
    st.markdown(
        "1. Choose the model path and tenure horizons.\n"
        "2. Complete the single-employee form or upload a CSV.\n"
        "3. Review the risk trajectory, policy guidance, and download results."
    )
    st.divider()
    st.header("Prediction settings")
    model_path_input = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
    tenure_horizons = st.multiselect(
        "Tenure horizons (years)",
        options=[1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
        default=list(DEFAULT_TENURE_HORIZONS),
    )
    tenure_horizons = tuple(sorted(set(tenure_horizons))) or DEFAULT_TENURE_HORIZONS
    st.caption("Predictions are generated for each selected horizon.")

    if best_model_name and metrics_payload:
        best_metrics = metrics_payload.get(best_model_name, {}).get("test", {})
        st.metric("Best model (test ROC-AUC)", f"{best_metrics.get('roc_auc', 0.0):.3f}", help=best_model_name)

    if metrics_table is not None:
        with st.expander("Model quality snapshot", expanded=False):
            formatted = (
                metrics_table.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
                .reset_index()
                .sort_values("model")
            )
            formatted[[col for col in formatted.columns if col != "model"]] = formatted[
                [col for col in formatted.columns if col != "model"]
            ].applymap(lambda v: f"{v:.3f}")
            st.dataframe(formatted, use_container_width=True)
    else:
        st.info("Run `make train` to generate evaluation metrics.")

# --- Main Panel ---
single_tab, batch_tab = st.tabs(["👤 Single employee", "📁 Batch CSV"])

with single_tab:
    st.subheader("Demographic, role, and engagement inputs")
    st.caption("All inputs are validated with the EmployeeRecord schema before inference.")
    with st.form("single_employee_form", clear_on_submit=False):
        record: Dict[str, Any] = {}
        form_errors = st.session_state.get("single_form_errors", {})
        _render_form_sections(record, categorical_options, numeric_defaults, date_defaults, form_errors)
        submitted = st.form_submit_button("🚀 Predict termination risk")

    if submitted:
        try:
            validated_record = EmployeeRecord.model_validate(record).normalized_payload()
            st.session_state["single_form_errors"] = {}
            estimator = load_prediction_model(model_path_input)
            risks_state = predict_tenure_risk(validated_record, horizons=tenure_horizons, model=estimator)
            st.success("Prediction complete")
            _display_prediction_results(risks_state, policy_config)
        except ValidationError as err:
            logger.warning("Validation failed for single record: %s", err)
            st.session_state["single_form_errors"] = _map_validation_errors(err)
            st.error("Please correct the highlighted fields before running prediction.")
        except Exception as exc:  # pragma: no cover - streamlit feedback
            logger.exception("Single-record prediction failed")
            st.error(f"Prediction failed: {exc}")

with batch_tab:
    st.subheader("Upload a CSV to score multiple employees")
    st.caption("The file must include the same columns as the single-employee form.")
    uploaded_file = st.file_uploader("Employee records (CSV)", type=["csv"], accept_multiple_files=False)

    if uploaded_file:
        try:
            records_df = pd.read_csv(uploaded_file)
        except Exception as exc:  # pragma: no cover - user input variability
            logger.exception("Failed to parse uploaded CSV")
            st.error(f"Could not parse file: {exc}")
            records_df = None

        if records_df is not None and records_df.empty:
            st.warning("The uploaded file did not contain any rows.")
        elif records_df is not None:
            st.markdown("**Preview (first 5 rows)**")
            st.dataframe(records_df.head(), use_container_width=True)
            try:
                estimator = load_prediction_model(model_path_input)
            except Exception as exc:  # pragma: no cover - streamlit feedback
                logger.exception("Failed to load model for batch predictions")
                st.error(f"Could not load model: {exc}")
            else:
                batch_results: List[pd.DataFrame] = []
                progress_bar = st.progress(0)
                for row_index, (_, row) in enumerate(records_df.iterrows(), start=1):
                    try:
                        normalized_row = EmployeeRecord.model_validate(row.to_dict()).normalized_payload()
                        risks = predict_tenure_risk(normalized_row, horizons=tenure_horizons, model=estimator)
                        risk_df = _tenure_risk_dataframe(risks, policy_config)
                        risk_df.insert(0, "record_index", row_index)
                        batch_results.append(risk_df)
                    except ValidationError as err:
                        messages = ", ".join(
                            f"{issue.get('loc', ['record'])[0]}: {issue.get('msg')}" for issue in err.errors()
                        )
                        st.warning(f"Row {row_index} failed validation and was skipped. Details: {messages}")
                    except Exception as exc:  # pragma: no cover - streamlit feedback
                        logger.warning("Prediction failed for row %d", row_index)
                        st.warning(f"Row {row_index} could not be scored: {exc}")
                    progress_bar.progress(row_index / len(records_df))

                if batch_results:
                    batch_df = pd.concat(batch_results, ignore_index=True)
                    preview_df = batch_df.copy()
                    preview_df["termination_probability"] = preview_df["termination_probability"].map(
                        _format_percentage
                    )
                    preview_df["confidence"] = preview_df["confidence"].map(_format_percentage)
                    st.success(f"Generated predictions for {preview_df['record_index'].nunique()} employees.")
                    st.dataframe(preview_df, use_container_width=True)

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download predictions.csv",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No valid predictions could be generated from the uploaded data.")

st.divider()
st.caption(
    "⚠️ This tool is for educational and demonstration purposes only. Do not use it for real HR decisions without"
    " rigorous validation, governance, and employee consent."
)
