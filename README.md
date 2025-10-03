# Will I Be Fired? – Termination Prediction Models

This project provides a reproducible workflow for predicting whether an employee will be terminated using the `HRDataset_v14.csv`
dataset. It includes reusable feature-engineering utilities, a training script that evaluates several classifiers under
cross-validation, persistable metrics, and an interactive inference helper that estimates termination risk at different tenure
horizons.


## Non-programmer handbook
If you are new to data science or coding, start with the [step-by-step non-programmer guide](docs/non_programmer_guide.md). It explains every concept, the full modelling pipeline, and how to run and understand the tools in plain language.

## Dataset overview
- **Records:** 311 employees
- **Target:** `Termd` (1 = terminated)
- **Class balance:** 104 terminated vs. 207 active employees (roughly 1:2)

### Feature engineering highlights
Feature preparation is centralised in [`src/feature_engineering.py`](src/feature_engineering.py):

- Date columns (`DateofHire`, `DateofTermination`, `DOB`, `LastPerformanceReview_Date`) are parsed with robust fallback logic.
- Temporal attributes are derived, including tenure in years, employee age, and time since the last performance review.
- Identifier and leakage-prone fields (IDs, manager names, termination reasons, etc.) are removed before modelling.
- Missing numeric features are median-imputed and scaled; categorical features are imputed with the modal value and one-hot
  encoded inside the modelling pipeline.
- Random over-sampling is applied inside the pipeline to mitigate the target imbalance without leaking validation data.

## Reproducing the training run

```bash
pip install -r requirements.txt
python src/train_model.py
```

The script trains logistic regression, random forest, and gradient boosting classifiers using a 5-fold stratified
cross-validation grid search. It evaluates each best estimator on held-out validation and test partitions and produces:

- The best-performing pipeline saved to `models/best_model.joblib` (ignored by Git by default—use Git LFS if you plan to commit
  model binaries).
- A comprehensive metrics report saved to [`reports/metrics.json`](reports/metrics.json).

Sample console output:

```
Best model: random_forest
Validation metrics: {
  "accuracy": 0.9787,
  "precision": 1.0,
  "recall": 0.9333,
  "roc_auc": 1.0
}
Test metrics: {
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "roc_auc": 1.0
}
```

## Inference utilities

### Python API
Use [`src/inference.py`](src/inference.py) to reload the persisted pipeline and generate predictions. The helpers accept raw
employee records (dicts, Series, or DataFrames) and handle the necessary feature engineering before calling the estimator.

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

### Command line interface
[`src/predict_cli.py`](src/predict_cli.py) provides an interactive prompt or JSON-driven workflow for generating risk
assessments at 1, 2, and 5 years of tenure (or custom horizons):

```bash
python src/predict_cli.py --model models/best_model.joblib
# or
python src/predict_cli.py --employee-json sample_employee.json --horizons 1 3 5
```

The CLI reports both the estimated probability of termination and a confidence score (the model's certainty in its prediction)
for each requested tenure horizon.

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

## Model performance summary

Validation and test metrics (best model chosen by validation ROC-AUC):

| Model | Split | Accuracy | Precision | Recall | ROC-AUC |
| ----- | ----- | -------- | --------- | ------ | ------- |
| Logistic Regression | Validation | 0.915 | 1.000 | 0.733 | 0.954 |
| Logistic Regression | Test | 0.979 | 1.000 | 0.938 | 0.996 |
| Random Forest (selected) | Validation | 0.979 | 1.000 | 0.933 | 1.000 |
| Random Forest (selected) | Test | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | Validation | 1.000 | 1.000 | 1.000 | 1.000 |
| Gradient Boosting | Test | 1.000 | 1.000 | 1.000 | 1.000 |

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

## Repository structure
- `HRDataset_v14.csv` – source dataset
- `models/` – persisted model pipeline artefacts (ignored by Git; regenerate with the training script)
- `reports/metrics.json` – validation/test metrics for each trained model
- `src/feature_engineering.py` – shared feature engineering utilities
- `src/inference.py` – reusable inference helpers, including tenure-based risk estimation
- `src/gui_app.py` – Streamlit-powered dashboard for interactive and batch predictions
- `src/logging_utils.py` – centralised logging helpers used across scripts
- `src/predict_cli.py` – interactive and file-driven CLI for risk predictions
- `src/train_model.py` – training and evaluation script
- `README.md` – project documentation (this file)
