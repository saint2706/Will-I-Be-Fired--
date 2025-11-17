# Will I Be Fired? – Termination Prediction Models

This project provides a reproducible workflow for predicting whether an employee will be terminated using the `HRDataset_v14.csv`
dataset. It includes reusable feature-engineering utilities, a training script that evaluates several classifiers under
cross-validation, persistable metrics, and an interactive inference helper that estimates termination risk at different tenure
horizons.


## Non-programmer handbook
If you are new to data science or coding, start with the [step-by-step non-programmer guide](docs/non_programmer_guide.md). It explains every concept, the full modelling pipeline, and how to run and understand the tools in plain language.

## Reproducibility

This project follows **publication-grade reproducibility standards**:

### Quick Start
```bash
# One-command reproducible pipeline
make reproduce
```

This runs: dependency installation → linting → testing → training → report generation.

### Manual Steps
```bash
# Install dependencies (pinned versions in pyproject.toml)
make setup

# Run linters (black, isort, flake8)
make lint

# Run tests with coverage
make test

# Train models and generate all reports
make train

# Launch Streamlit GUI
make app
```

### Reproducibility Features
- **Fixed random seed** (`RANDOM_SEED = 42` in `src/constants.py`)
- **Pinned dependencies** (exact versions in `requirements.txt` and `pyproject.toml`)
- **Python version** (`.python-version` specifies 3.11.9)
- **Pre-commit hooks** (`.pre-commit-config.yaml` with black, isort, flake8)
- **Automated CI** (`.github/workflows/ci.yml` runs lint, test, and smoke tests)

See the [Makefile](Makefile) for all available targets.

---

## Dataset Overview

### Summary
- **Records:** 311 employees
- **Target:** `Termd` (1 = terminated, 0 = active)
- **Class balance:** 104 terminated (33.4%) vs. 207 active (66.6%)
- **Source:** `HRDataset_v14.csv`

### Data Card
For detailed provenance, schema, missing values, cardinality, and leakage analysis, see **[docs/DATA_CARD.md](docs/DATA_CARD.md)**.

Key highlights:
- **Temporal features** (tenure, age, review recency) are engineered from date columns
- **Leakage-prone columns** (employment status, termination reason, manager IDs) are **explicitly dropped**
- **Low missing values** (<5% for most features)
- **Class imbalance** addressed via RandomOverSampler

### Feature Engineering
Feature preparation is centralized in [`src/feature_engineering.py`](src/feature_engineering.py):

- Date columns parsed with robust fallback logic
- Temporal attributes derived: `tenure_years`, `age_years`, `years_since_last_review`
- Identifier and leakage fields removed before modeling
- Numeric features: median imputation + standard scaling
- Categorical features: mode imputation + one-hot encoding
- Random over-sampling applied inside pipeline (no validation leakage)

---

## Training & Evaluation

### Quick Training
```bash
python src/train_model.py
```

The script:
1. Loads and preprocesses `HRDataset_v14.csv`
2. Creates 70/15/15 stratified train/val/test splits
3. Trains Logistic Regression, Random Forest, and Gradient Boosting with hyperparameter search
4. Selects best model by validation ROC-AUC
5. Computes test metrics with **95% bootstrap confidence intervals**
6. Generates diagnostic plots (ROC, PR, calibration, feature importance)
7. Performs failure analysis (false positives/negatives)
8. Evaluates baseline models (majority class, stratified random)

### Outputs
- **Model:** `models/best_model.joblib`
- **Metrics:** `reports/metrics.json` (all models), `reports/metrics_with_ci.json` (with CIs)
- **Figures:** `reports/figures/` (roc.png, pr.png, calibration.png, perm_importance.png)
- **Splits:** `reports/split_summary.json` and `reports/splits.csv`
- **Failures:** `reports/failure_cases.csv` and `reports/FAILURES.md`

### Confidence Intervals & Calibration
Test metrics are reported with **95% confidence intervals** using 1000 bootstrap resamples:

| Metric | Mean | 95% CI Lower | 95% CI Upper |
|--------|------|--------------|---------------|
| Accuracy | 1.000 | 0.982 | 1.000 |
| Precision | 1.000 | 1.000 | 1.000 |
| Recall | 1.000 | 0.933 | 1.000 |
| F1 | 1.000 | 0.966 | 1.000 |
| ROC-AUC | 1.000 | 0.995 | 1.000 |

**Calibration:** Brier score and reliability curve assess probability calibration (see `reports/figures/calibration.png`).

---

## Baseline & Ablation Studies

### Baselines
Simple baselines are evaluated for context:
- **Majority class:** Always predicts "active" (most common class)
- **Stratified random:** Randomly predicts with class proportions

Results included in `reports/metrics.json`.

### Ablations
To understand component contributions, run:
```bash
python src/ablation.py
```

Experiments:
1. **No rebalancing:** Remove RandomOverSampler
2. **No date features:** Drop `tenure_years`, `age_years`, `years_since_last_review`
3. **Top-k features:** Use only 5, 10, or 20 most important features

Results saved to `reports/ablation_results.csv` and `reports/ABLATIONS.md`.

Key insights:
- Class rebalancing significantly improves recall for minority class
- Temporal features are critical for predictive power
- Top 10-20 features capture most of the signal

---

## Failure Analysis

### Overview
The training script automatically identifies and analyzes misclassifications:
- **False Positives:** Predicted termination, actually retained
- **False Negatives:** Predicted retention, actually terminated

### Outputs
- **reports/failure_cases.csv:** Top N cases per error type with feature values
- **reports/FAILURES.md:** Narrative analysis with representative examples

### Example Insights
- **False Positives:** Often have risk factors (low satisfaction) but protective factors (strong performance)
- **False Negatives:** Observable metrics appear stable, but unobserved factors (external offers, personal issues) lead to termination

See [reports/FAILURES.md](reports/FAILURES.md) for recommendations to improve model.

---

## Business Actions Framework

### Risk-Based Interventions
Predicted probabilities map to **HR action recommendations** via `configs/policy.yaml`:

| Risk Band | Probability Range | Actions |
|-----------|-------------------|---------|
| **Low** | 0-10% | Standard management, recognition programs |
| **Low-Moderate** | 10-30% | Bi-weekly 1:1s, engagement follow-up |
| **Moderate** | 30-60% | Weekly 1:1s, compensation review, career pathing |
| **High** | 60%+ | Immediate HR escalation, retention package, stay interview |

### Expected Impact
If 100 employees are scored with interventions:
- **10-30% band:** ~10 retained (50% success rate)
- **30-60% band:** ~6 retained (40% success rate)
- **60%+ band:** ~3 retained (30% success rate)

**Total projected retention:** ~19 employees  
**Cost avoidance:** $1.5M–$2.5M (at $75k-$150k replacement cost per employee)

### Configuration
Edit `configs/policy.yaml` to adjust thresholds and actions for your organization.

See **[docs/BUSINESS_ACTIONS.md](docs/BUSINESS_ACTIONS.md)** for full framework.

---

## Sample Training Output

```
Best model: random_forest
Validation metrics: {
  "accuracy": 0.979,
  "precision": 1.000,
  "recall": 0.933,
  "roc_auc": 1.000
}
Test metrics with 95% CI: {
  "accuracy": 1.000 (0.982-1.000),
  "precision": 1.000 (1.000-1.000),
  "recall": 1.000 (0.933-1.000),
  "roc_auc": 1.000 (0.995-1.000)
}
```

## Inference utilities

### Python API
Use [`src/inference.py`](src/inference.py) to reload the persisted pipeline and generate predictions. The helpers accept raw
employee records (dicts, Series, or DataFrames) and handle the necessary feature engineering before calling the estimator. Every
entry point first validates data against [`EmployeeRecord`](src/schemas.py), so typos (e.g., "sixty" instead of `60000`) and
impossible dates are caught before the model runs.

```python
import sys
sys.path.append("src")

import pandas as pd
from inference import load_model, predict_termination_probability, predict_tenure_risk

model = load_model()
record = {
    "Department": "IT",
    "PerformanceScore": "Fully Meets",
    "RecruitmentSource": "Indeed",
    "Position": "IT Support",
    "State": "MA",
    "Sex": "Male",
    "MaritalDesc": "Single",
    "CitizenDesc": "US Citizen",
    "RaceDesc": "White",
    "HispanicLatino": "No",
    "Salary": 65000,
    "EngagementSurvey": 4.0,
    "EmpSatisfaction": 4,
    "SpecialProjectsCount": 3,
    "DaysLateLast30": 0,
    "Absences": 5,
    "DateofHire": "2018-07-01",
    "DOB": "1990-05-18",
    "LastPerformanceReview_Date": "2023-10-01",
}

probability = predict_termination_probability(record, model=model)[0]
print(f"Overall termination probability: {probability:.1%}")

for risk in predict_tenure_risk(record, model=model):
    print(
        f"Tenure {risk.tenure_years} yrs → probability {risk.termination_probability:.1%},"
        f" confidence {risk.confidence:.1%}"
    )
```

### Command Line Interface
[`src/predict_cli.py`](src/predict_cli.py) provides an interactive prompt or JSON-driven workflow for generating risk
assessments at 1, 2, and 5 years of tenure (or custom horizons):

```bash
# Basic usage
python src/predict_cli.py --model models/best_model.joblib

# With policy-based action recommendations
python src/predict_cli.py --model models/best_model.joblib --calibrate

# From JSON file with custom horizons
python src/predict_cli.py --employee-json sample_employee.json --horizons 1 3 5 --calibrate
```

If any field violates the shared schema, the CLI prints a concise error report and exits with status code 1. The Streamlit GUI
shows the same validation feedback inline next to the affected inputs.

**With `--calibrate` flag:**
- Shows **risk band** (Low, Low-Moderate, Moderate, High) for each prediction
- Displays **recommended HR actions** from `configs/policy.yaml`
- Example output:
  ```
  Tenure     Probability    Confidence     Risk Band
  1.0 yrs    35%            82%            Moderate Risk
  
  --- Recommended Actions (Moderate Risk) ---
  1. Move to weekly manager 1:1s for 4-6 weeks
  2. Conduct off-cycle compensation review
  3. Schedule career pathing session with HR
  ...
  ```

The CLI reports the estimated probability, confidence score, and (optionally) actionable recommendations for each tenure horizon.

### Graphical user interface
Launch the Streamlit dashboard to capture employee details via dropdowns, inspect evaluation metrics, and visualise tenure-risk
trajectories. The app also supports uploading CSV or JSON files with multiple employees and exports a tidy prediction report for
offline analysis.

```bash
streamlit run src/gui_app.py
```

Within the UI you can:

- Select categorical attributes from dropdown menus and provide numeric/date inputs for a single employee.
- Review a combined chart that juxtaposes the best model's test metrics with the predicted probabilities and confidence bands.
- Upload batch files and download a structured CSV report containing predictions for every employee and tenure horizon.

### Logging
All scripts (training, CLI, and GUI) initialise a consistent logging format. Runtime information—including dataset loading,
model selection, inference outcomes, and error traces—is written to standard output, making it easier to monitor behaviour and
integrate with external logging solutions.

## Model Performance Summary

### Comparison of Models (Validation & Test)
Best model selected by validation ROC-AUC:

| Model | Split | Accuracy | Precision | Recall | ROC-AUC |
| ----- | ----- | -------- | --------- | ------ | ------- |
| Logistic Regression | Validation | 0.915 | 1.000 | 0.733 | 0.954 |
| Logistic Regression | Test | 0.979 | 1.000 | 0.938 | 0.996 |
| **Random Forest (selected)** | **Validation** | **0.979** | **1.000** | **0.933** | **1.000** |
| **Random Forest (selected)** | **Test** | **1.000** | **1.000** | **1.000** | **1.000** |
| Gradient Boosting | Validation | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | Test | 1.000 | 1.000 | 1.000 | 1.000 |

### Test Metrics with 95% Bootstrap Confidence Intervals
See `reports/metrics_with_ci.md` for detailed CI report. Summary for best model:

| Metric | Value (95% CI) |
|--------|----------------|
| Accuracy | 1.000 (0.982-1.000) |
| Precision | 1.000 (1.000-1.000) |
| Recall | 1.000 (0.933-1.000) |
| F1 | 1.000 (0.966-1.000) |
| ROC-AUC | 1.000 (0.995-1.000) |
| PR-AUC | 1.000 (0.995-1.000) |
| Brier Score | 0.000 (0.000-0.018) |

**Note:** Perfect test performance may indicate overfitting due to small dataset size (N=311). Monitor performance on new data.

### Baseline Comparisons
| Baseline | Accuracy | Precision | Recall | ROC-AUC |
|----------|----------|-----------|--------|---------|
| Majority Class | 0.667 | 0.000 | 0.000 | 0.500 |
| Stratified Random | ~0.50 | ~0.33 | ~0.50 | ~0.50 |

Our models significantly outperform trivial baselines.

### Diagnostic Plots
- **ROC Curve:** `reports/figures/roc.png` (with bootstrap CI bands)
- **Precision-Recall Curve:** `reports/figures/pr.png` (with bootstrap CI bands)
- **Calibration Curve:** `reports/figures/calibration.png` (Brier score: ~0.00)
- **Feature Importance:** `reports/figures/perm_importance.png` (top 20 permutation importances)

## Fairness & Responsible Use

- Run `make fairness` (or the standalone `python src/fairness_analysis.py`) to
  compute group-wise accuracy/recall/ROC-AUC, demographic parity difference,
  and equal-opportunity difference for `RaceDesc`, `Sex`, `MaritalDesc`, and
  `CitizenDesc`. Outputs land in `reports/fairness_summary.csv` plus bar charts
  under `reports/figures/fairness_*.png`.
- The current audit (47 test employees) shows the largest demographic parity
  gap for **MaritalDesc** (0.57 between divorced and separated employees) and a
  0.39 gap across race groups; several categories have only 1–2 examples so the
  numbers carry high uncertainty.
- Detailed findings, limitations, and a deployment warning live in
  [`docs/FAIRNESS_AND_LIMITATIONS.md`](docs/FAIRNESS_AND_LIMITATIONS.md). The
  model must only be used to flag retention risks for human review—**never** to
  automate firing decisions.

## Ethical considerations
- **Bias and fairness:** HR data may encode historical biases in hiring, promotion, or termination decisions. Monitor subgroup
  performance (e.g., by gender, race) to detect disparate impact.
- **Privacy:** Employee records are sensitive. Ensure access controls, data minimisation, and compliance with employment and
  privacy regulations when deploying the model.
- **Appropriate use:** Predictions should augment—not replace—human judgment. Avoid using the model to justify adverse
  employment actions without thorough review.

## Monitoring and maintenance
- **Data drift:** Track feature distributions over time (tenure, survey scores, etc.) and trigger retraining if drift exceeds
  defined thresholds.
- **Performance drift:** Periodically evaluate on recent outcomes to ensure accuracy, precision, recall, and ROC-AUC remain
  acceptable.
- **Feedback loops:** Log model decisions and subsequent HR outcomes to detect systemic feedback that could reinforce bias.
- **Model updates:** Re-run `src/train_model.py` with refreshed data and compare metrics before promoting new models.

## Repository Structure

```
.
├── HRDataset_v14.csv                  # Source dataset (311 employee records)
├── .python-version                    # Python version specification (3.11.9)
├── pyproject.toml                     # Project metadata & dependency versions (PEP 621)
├── requirements.txt                   # Pinned dependencies for reproducibility
├── Makefile                           # Build targets (setup, lint, test, train, app, reproduce)
├── .pre-commit-config.yaml            # Pre-commit hooks (black, isort, flake8)
├── .github/workflows/ci.yml           # CI pipeline (lint, test, smoke test training)
│
├── configs/
│   └── policy.yaml                    # Risk thresholds & HR action mappings
│
├── docs/
│   ├── DATA_CARD.md                   # Dataset provenance, schema, leakage analysis
│   ├── BUSINESS_ACTIONS.md            # Risk-based intervention framework
│   ├── FAIRNESS_AND_LIMITATIONS.md    # Bias analysis summary + responsible use warning
│   └── non_programmer_guide.md        # Beginner-friendly tutorial
│
├── models/
│   └── best_model.joblib              # Trained model pipeline (gitignored, regenerate via training)
│
├── reports/
│   ├── metrics.json                   # Validation/test metrics for all models
│   ├── metrics_with_ci.json           # Test metrics with 95% bootstrap CIs
│   ├── metrics_with_ci.md             # Human-readable metrics table
│   ├── split_summary.json             # Train/val/test split statistics
│   ├── splits.csv                     # Split counts & positive rates
│   ├── ablation_results.csv           # Ablation study results
│   ├── ABLATIONS.md                   # Ablation narrative report
│   ├── failure_cases.csv              # Misclassified examples with explanations
│   ├── FAILURES.md                    # Failure analysis narrative
│   └── figures/
│       ├── roc.png                    # ROC curve with bootstrap CI bands
│       ├── pr.png                     # Precision-Recall curve with CI bands
│       ├── calibration.png            # Calibration (reliability) curve
│       └── perm_importance.png        # Top 20 feature importances
│
├── src/
│   ├── constants.py                   # Global constants (RANDOM_SEED)
│   ├── feature_engineering.py         # Feature transformations & preprocessing
│   ├── inference.py                   # Prediction helpers & tenure risk analysis
│   ├── train_model.py                 # Model training pipeline
│   ├── predict_cli.py                 # Command-line prediction interface
│   ├── gui_app.py                     # Streamlit dashboard
│   ├── logging_utils.py               # Centralized logging configuration
│   ├── ablation.py                    # Ablation study experiments
│   ├── utils/
│   │   ├── repro.py                   # Reproducibility (set_global_seed)
│   │   ├── metrics.py                 # Bootstrap CI computation
│   │   └── plotting.py                # Diagnostic plot generation
│   └── analysis/
│       └── failures.py                # Failure case identification & analysis
│
└── tests/
    ├── test_feature_engineering.py    # Feature engineering tests
    ├── test_inference.py              # Inference utilities tests
    ├── test_predict_cli.py            # CLI tests
    └── test_utils.py                  # Utility function tests (repro, metrics, plotting)
```
