# Non-programmer guide to the "Will I Be Fired?" project

This document was written for readers who do not code or work with machine learning every day. It explains the data, the math,
the training workflow, and every way you can use the project tools. If you only want the highlights, read the overview. If you
want to understand the full pipeline and the meaning of every metric, continue through the remaining sections.

---

## 1. Project overview in plain language

- **Goal:** Use past HR records to estimate the chance that an employee will be terminated in the future.
- **Dataset:** A table called `HRDataset_v14.csv` where each row is an employee and each column describes something about them
  (department, salary, survey scores, etc.). One special column, `Termd`, marks whether the employee eventually left.
- **Why predictions are useful:** HR teams can use the probabilities as early warning signals and pair them with supportive
  interventions. The model must *not* replace human judgement; it should help start conversations.

---

## What data do I need to provide?

All inference tools (Python API, CLI, and Streamlit dashboard) share the exact same input schema defined in `src/schemas.py`.
That schema describes what HR partners must supply before any prediction runs.

### Categorical descriptors

Provide the latest value for each text field. Leave no blanks—if something truly does not apply, enter `Unknown` so the
validation layer can route it consistently.

- Department
- PerformanceScore (e.g., "Fully Meets", "Needs Improvement")
- RecruitmentSource ("Indeed", "LinkedIn", internal referral, etc.)
- Position (job title)
- State (two-letter postal code)
- Sex
- MaritalDesc ("Single", "Married", ...)
- CitizenDesc (citizenship/visa status)
- RaceDesc
- HispanicLatino ("Yes" or "No")

### Numeric and survey inputs

| Field | What to enter | Constraints | Default if omitted |
| --- | --- | --- | --- |
| Salary | Annual compensation in USD | 0 – 1,000,000 | 65,000 |
| EngagementSurvey | Engagement score (0–5) rounded to 2 decimals | 0 – 5 | 4.0 |
| EmpSatisfaction | Whole-number satisfaction rating | 1 – 5 | 4 |
| SpecialProjectsCount | Count of active special projects | 0 – 50 | 3 |
| DaysLateLast30 | Total late days during the last 30-day window | 0 – 30 | 0 |
| Absences | Calendar days of absence over the past year | 0 – 365 | 5 |

The defaults are identical to the CLI prompts; they help with exploratory runs but you should replace them with real data.

### Critical dates

- **DateofHire** – First day the employee was paid. Must fall between 1900-01-01 and 2100-12-31.
- **DOB** – Date of birth. Must also be within the allowed range *and* occur before the hire date.
- **LastPerformanceReview_Date** – When the most recent formal review concluded. Must be on/after the hire date.

Dates can be typed as `YYYY-MM-DD` strings or selected from the GUI calendar. The validation layer normalizes every valid input
to ISO 8601 format so downstream code can safely parse it.

If any field is missing, mistyped (for example, "sixty" instead of `60_000`), or contains an impossible date, the interface will
show an actionable error message and refuse to run inference until the record conforms to this schema.

---

## 2. How the data becomes predictions

The workflow follows nine repeatable stages. You can re-run them with the scripts in the `src/` folder or by using the `make reproduce` command (see "Option A: One-Command Reproducible Pipeline" in Section 5).

1. **Load the data** – Read the CSV file into memory using pandas (a spreadsheet-like Python library). Every column is kept in
a structured format so later steps can understand whether a value is text, a number, or a date.
2. **Clean and enrich the features** – Convert human-readable dates into machine-friendly numbers (for example tenure in years
   or how long since the last review), drop columns that could leak the true outcome (for example the literal termination date),
   fill in missing values, and convert words into numbers so algorithms can work with them. The code that does this lives in
   [`src/feature_engineering.py`](../src/feature_engineering.py).
3. **Split the dataset** – Divide the rows into three non-overlapping pieces:
   - **Training set (70%)** – What the algorithms learn from.
   - **Validation set (15%)** – Used during tuning to choose the best algorithm without touching the final test.
   - **Test set (15%)** – Held back until the end so we know how the chosen model behaves on truly unseen people.
   The splitting keeps the same proportion of terminated vs. active employees in each subset (this is called *stratification*).
4. **Train multiple models** – Three popular machine-learning approaches are tried:
   - **Logistic regression** – A simple formula that estimates the log-odds of termination as a weighted sum of the input
     features.
   - **Random forest** – A collection of decision trees that vote together. Each tree looks at random subsets of features to
     avoid overfitting.
   - **Gradient boosting** – Builds trees one at a time, each focusing on correcting the mistakes of the previous ones.
   
   Additionally, two baseline models are evaluated for comparison:
   - **Majority class baseline** – Always predicts the most common outcome (no termination).
   - **Stratified random baseline** – Makes random predictions matching the overall class distribution.
5. **Evaluate and select the winner** – Every model is assessed with accuracy, precision, recall, and ROC-AUC (explained in
   Section 4). The model with the best validation ROC-AUC is kept, then tested one final time on the test set to estimate
   real-world performance. Test metrics are reported with **95% bootstrap confidence intervals** to quantify uncertainty.
6. **Generate diagnostic plots** – The system automatically creates visual reports showing:
   - **ROC curve** with confidence bands – Shows how well the model separates terminated from active employees.
   - **Precision-Recall curve** with confidence bands – Useful for understanding performance on the minority class.
   - **Calibration curve** – Checks if the predicted probabilities match actual outcomes.
   - **Feature importance chart** – Identifies which employee attributes matter most for predictions.
7. **Analyze failures** – The system examines incorrect predictions to understand patterns:
   - **False positives** – Employees predicted to be terminated who actually stayed.
   - **False negatives** – Employees predicted to stay who actually left.
   This analysis helps identify model limitations and areas for improvement.
8. **Run ablation studies (optional)** – These experiments measure what happens when you remove specific components:
   - Training without class rebalancing
   - Training without date-derived features (tenure, age, etc.)
   - Training with only the top most important features
9. **Save the pipeline** – The winning model and every preprocessing step are bundled together into a single pipeline object and
   stored as `models/best_model.joblib`. When you load it later, it remembers how to clean new records in the same way.

The entire routine is automated in [`src/train_model.py`](../src/train_model.py). Running that script from the command line
recreates every experiment and produces comprehensive reports for transparency (see Section 6 for details on all outputs).

---

## 3. What happens during feature engineering

Below is a checklist of the transformations performed on every incoming record, whether it is used for training or for
prediction.

1. **Parsing dates** – `DateofHire`, `DOB`, `DateofTermination`, and `LastPerformanceReview_Date` are converted from text into
   actual calendar dates. Invalid strings are handled gracefully so the program never crashes.
2. **Derived numbers** – The parsed dates are used to compute:
   - **Tenure (years):** `today - DateofHire`.
   - **Age (years):** `today - DOB`.
   - **Time since last review (days):** `today - LastPerformanceReview_Date`.
   - **Tenure at termination:** Only used internally to flag data leakage and is removed before modelling.
3. **Dropping leakage fields** – Anything that reveals the future (like `TermReason` or `DateofTermination`) is excluded.
4. **Handling categorical text** – Columns like `Department` or `PerformanceScore` are treated as categorical variables and
   turned into binary indicators through one-hot encoding (e.g. `Department_HR` equals 1 if the person works in HR).
5. **Handling numbers** – Numeric columns are filled with their median value when missing and then standardised (subtract the
   mean and divide by the standard deviation) so different scales do not overpower the learning algorithm.
6. **Balancing the classes** – Because there are fewer terminated employees than active ones, the pipeline uses *RandomOverSampler*
   from the `imblearn` package to duplicate minority examples inside each training split. This avoids leaking information from
   the validation or test sets.

---

## 4. Metrics explained without jargon

Every metric is calculated twice: once on the validation set and once on the test set. High scores across both are a good sign. For the test set, all metrics include **95% confidence intervals** computed using bootstrap resampling (explained below).

### Core Performance Metrics

- **Accuracy** – The share of all predictions the model gets right. Formula: `(true positives + true negatives) / all cases`.
  Works best when both classes are balanced; for imbalanced data it can be misleading.
- **Precision** – When the model predicts someone will be terminated, precision measures how often that is actually true.
  Formula: `true positives / (true positives + false positives)`.
- **Recall (also called sensitivity)** – Out of everyone who really gets terminated, recall tells you how many the model caught.
  Formula: `true positives / (true positives + false negatives)`.
- **F1 Score** – The harmonic mean of precision and recall. It balances both metrics into a single number, which is useful
  when you care equally about catching all terminations and avoiding false alarms.
- **ROC-AUC** – Stands for *Receiver Operating Characteristic – Area Under the Curve*. The model produces a probability between
  0 and 1. ROC-AUC looks at every possible probability threshold and measures how well the model separates terminated from active
  employees. A perfect score is 1.0; 0.5 means guessing.

### Confidence Intervals and Calibration

- **Bootstrap confidence intervals** – Because the test set is small, the exact metric values have some uncertainty. To quantify
  this, the system creates 1,000 random resamples of the test data (with replacement), computes the metric on each resample,
  and reports the middle 95% of results. For example, "Accuracy: 1.000 (0.982-1.000)" means the accuracy is 1.0, but if we
  tested on a slightly different set of employees, it would likely fall between 98.2% and 100%.
- **Calibration metrics** – The Brier score and calibration curve measure if predicted probabilities are trustworthy. A well-
  calibrated model that says "30% chance of termination" should be right about 30% of the time across many such predictions.
- **Confidence score in predictions** – When you request future risks at specific tenure milestones (1, 2, 5 years), the tool
  reports both the predicted probability and a confidence value. The confidence is derived from the model's probability output
  by measuring how far it sits from uncertainty (0.5). The further away, the more confident the model is about that prediction.

---

## 5. Step-by-step: running the project without writing code

Follow these steps on macOS, Windows (with PowerShell), or Linux. Replace `<path>` with the actual folder where you stored the
repository.

1. **Install Python 3.11 or newer** (the project requires Python 3.11+, as specified in `pyproject.toml`. It is tested with Python 3.11.9 specified in `.python-version`, and should work with any Python 3.11.x or later versions).
2. **Open a terminal** (Command Prompt, PowerShell, or Terminal app).
3. **Move into the project folder:**

   ```bash
   cd <path>/Will-I-Be-Fired--
   ```

4. **Choose your workflow:**

   ### Option A: One-Command Reproducible Pipeline (Recommended)
   
   Run everything automatically with a single command:
   
   ```bash
   make reproduce
   ```
   
   This command will:
   - Install all dependencies (including development tools for code quality)
   - Run code quality checks (linters)
   - Run all tests
   - Train the model from scratch
   - Generate all reports and diagnostic plots
   
   The entire process takes 5-10 minutes. When complete, check the `reports/` folder for all outputs.

   ### Option B: Manual Step-by-Step

   If you prefer more control, run each stage separately:

   a. **Install the dependencies:**

      ```bash
      make setup
      ```
      
      Or manually:
      
      ```bash
      pip install -r requirements.txt
      ```

   b. **Run code quality checks (optional):**

      ```bash
      make lint
      ```
      
      This checks the code for style consistency. If you want to auto-fix formatting issues:
      
      ```bash
      make format
      ```

   c. **Run tests (optional):**

      ```bash
      make test
      ```

   d. **Train or retrain the model:**

      ```bash
      make train
      ```
      
      Or manually:

      ```bash
      python src/train_model.py
      ```

      This command prints progress logs, saves the best model to `models/best_model.joblib`, and writes detailed metrics and
      reports to the `reports/` folder (see Section 6 for details on outputs).

5. **Run ablation studies (optional):**

   To understand which components contribute most to model performance:

   ```bash
   python src/ablation.py
   ```
   
   This generates `reports/ablation_results.csv` and `reports/ABLATIONS.md` showing performance with various features removed.

6. **Choose how you want to make predictions:**
   - **Interactive command line:**

     ```bash
     python src/predict_cli.py --model models/best_model.joblib
     ```

     The program will ask you questions one-by-one. You can also pass a JSON file containing employee records:

     ```bash
     python src/predict_cli.py --employee-json my_team.json --model models/best_model.joblib --horizons 1 2 5
     ```

   - **Streamlit graphical interface:**

     ```bash
     make app
     ```
     
     Or manually:

     ```bash
     streamlit run src/gui_app.py
     ```

     A browser tab opens where you can select values from dropdown menus, upload CSV/JSON files with many employees, and download
     a neatly formatted report of the predictions.

---

## 6. Understanding the outputs

### 6.1 Console and log messages

Every script writes structured logs like:

```log
2024-04-01 10:15:12 INFO  feature_engineering: Loaded 311 rows from HRDataset_v14.csv
```

- **INFO** messages describe progress.
- **WARNING/ERROR** messages highlight problems and always include suggestions to fix them.
All logs are timestamped so you can follow what happened in which order.

### 6.2 Training outputs (in the `reports/` folder)

After training completes, multiple files are generated to help you understand model performance:

#### Core Metrics Reports

- **`metrics.json`** – Contains performance metrics for all three models (logistic regression, random forest, gradient boosting)
  plus baseline comparisons. Each model has validation and test scores for accuracy, precision, recall, and ROC-AUC.

- **`metrics_with_ci.json`** – Test set metrics with **95% bootstrap confidence intervals**. Shows the point estimate plus
  lower and upper bounds for each metric.

- **`metrics_with_ci.md`** – A human-readable Markdown table of the confidence interval metrics, formatted like:
  ```
  | Metric   | Mean  | 95% CI Lower | 95% CI Upper |
  |----------|-------|--------------|---------------|
  | Accuracy | 1.000 | 0.982        | 1.000         |
  ```

#### Data Split Information

- **`split_summary.json`** – Records how the data was divided (70% train, 15% validation, 15% test) including exact counts
  and class distributions in each split.

- **`splits.csv`** – Lists which employee indices went into which split, useful for reproducibility and debugging.

#### Failure Analysis

- **`failure_cases.csv`** – A table of misclassified examples showing the employee's features, true outcome, predicted outcome,
  and predicted probability. Helps identify patterns in model errors.

- **`FAILURES.md`** – A narrative report analyzing false positives and false negatives, with representative examples and
  suggested improvements.

#### Diagnostic Plots (in `reports/figures/`)

All plots are saved as PNG images:

- **`roc.png`** – ROC curve with confidence bands showing the trade-off between true positive rate and false positive rate.
- **`pr.png`** – Precision-Recall curve with confidence bands, particularly useful for imbalanced datasets.
- **`calibration.png`** – Calibration curve comparing predicted probabilities to actual outcomes.
- **`perm_importance.png`** – Bar chart of the top 20 most important features based on permutation importance.

#### Ablation Study Results (generated by `python src/ablation.py`)

- **`ablation_results.csv`** – Performance metrics when different components are removed (e.g., no rebalancing, no date features).
- **`ABLATIONS.md`** – Narrative summary of ablation findings.

### 6.3 Prediction outputs

- **CLI output:** For each employee and each requested tenure horizon, you receive a line like:

  ```text
  Employee 1 — Tenure 2 years → termination probability 12.4% (confidence 76.0%)
  ```

- **Streamlit dashboard:** Displays a table with the probabilities and confidence levels, plus a chart showing how risk changes
  over time. The chart also reminds you of the model's overall accuracy, precision, recall, and ROC-AUC.
- **Downloaded report:** When you process multiple employees, the GUI offers a CSV download. Each row contains the original
  employee identifier (if provided), the tenure horizon, the probability, and the confidence.

To interpret these numbers:

- Probabilities close to **0%** indicate low risk; close to **100%** means high risk.
- Confidence above **75%** means the model was far away from uncertainty; numbers near **50%** should be treated with caution.
- Always combine the predictions with context from HR business partners and employee conversations.

### 6.4 Additional reference documentation

The project includes comprehensive documentation beyond this guide:

- **`docs/DATA_CARD.md`** – Detailed information about the dataset including provenance, schema, missing values, and potential
  biases. Read this to understand data quality and limitations.

- **`docs/BUSINESS_ACTIONS.md`** – A framework for translating risk scores into specific HR interventions. Maps probability
  ranges (0-10%, 10-30%, 30-60%, 60%+) to concrete actions like increased check-ins, compensation reviews, or retention
  packages. Also includes expected impact estimates.

- **`configs/policy.yaml`** – A configuration file where you can customize risk band thresholds and action recommendations for
  your organization.

---

## 7. Using predictions to take action

Prediction probabilities alone don't tell HR teams what to do. The project includes a **Business Actions Framework** that maps
risk scores to specific interventions.

### Risk bands and recommended actions

The framework divides predictions into four bands:

| Risk Band | Probability Range | Recommended Actions | Expected Retention Rate |
|-----------|-------------------|---------------------|-------------------------|
| **Low** | 0-10% | Standard management, recognition programs | ~95% naturally retained |
| **Low-Moderate** | 10-30% | Bi-weekly 1:1s, engagement survey follow-up | ~50% retained with intervention |
| **Moderate** | 30-60% | Weekly 1:1s, compensation review, career pathing, mentorship | ~40% retained with intervention |
| **High** | 60%+ | Immediate HR escalation, retention package, stay interview | ~30% retained with intervention |

### How to use the framework

1. **Generate predictions** for your employee population using the CLI or GUI tools.
2. **Group employees by risk band** using the probability thresholds above.
3. **Apply the recommended actions** for each band. See `docs/BUSINESS_ACTIONS.md` for detailed descriptions of each
   intervention.
4. **Track outcomes** by recording which employees received interventions and whether they stayed. This data can improve future
   models and refine which actions work best.

### Cost-benefit analysis

For a typical organization with 100 at-risk employees:
- **Intervention costs:** ~$50k-$100k (time for 1:1s, retention bonuses, etc.)
- **Expected retention:** 15-20 additional employees retained
- **Cost avoidance:** $1.5M-$2.5M at $75k-$150k replacement cost per employee

The framework is configurable via `configs/policy.yaml`, allowing you to adjust thresholds and actions for your organization's
context.

---

## 8. Advanced features: Ablation studies

Ablation studies help you understand which components of the model are essential. By removing features or techniques one at a
time, you can see how much each contributes to overall performance.

### Running ablation experiments

```bash
python src/ablation.py
```

This script runs three types of experiments:

1. **No class rebalancing** – Trains without RandomOverSampler to see how important balancing the dataset is for catching
   terminations (recall).
2. **No temporal features** – Removes `tenure_years`, `age_years`, and `years_since_last_review` to measure how much predictive
   power comes from time-based attributes.
3. **Top-k features only** – Uses only the 5, 10, or 20 most important features to understand if a simpler model would suffice.

### Interpreting results

Results are saved to `reports/ablation_results.csv` and `reports/ABLATIONS.md`. Typically, you'll find:

- **Class rebalancing is critical** for recall. Without it, the model tends to predict "no termination" too often.
- **Temporal features matter a lot.** Tenure and age are among the strongest predictors.
- **Diminishing returns with features.** The top 10-20 features capture most of the signal; adding all features provides only
  marginal improvements.

Use ablation results to make decisions about model complexity. If a simpler model (top 10 features, no rebalancing) performs
nearly as well, it may be easier to explain and maintain.

---

## 9. Ethics, responsible use, and monitoring

- **Bias checks:** Periodically review performance separately for different demographic groups. The `failure_cases.csv` can
  help identify if certain groups are disproportionately misclassified.
- **Explainability:** Use the feature importance plot (`reports/figures/perm_importance.png`) to see which attributes drive
  predictions. For individual predictions, you can examine which feature values are unusual compared to the training data.
- **Data privacy:** Keep the dataset and prediction reports in secure storage with strict access controls. Employee termination
  risk is sensitive information and should be handled according to your organization's data governance policies.
- **Actionable insights:** Review `docs/BUSINESS_ACTIONS.md` for guidance on translating predictions into supportive HR
  interventions. The model should inform conversations, not replace human judgment.
- **Retraining cadence:** Schedule a review (for example every quarter). If the input data distribution changes or if metrics on
  fresh data degrade, re-run `make train` or `python src/train_model.py` to produce an updated model. The project includes
  automated CI/CD workflows (`.github/workflows/retrain-model.yml`) that can retrain on a schedule.
- **Model limitations:** The failure analysis reports (`FAILURES.md`) highlight cases where the model struggles. Use these
  insights to understand when predictions are less reliable and where additional data or features might help.

---

## 10. Troubleshooting checklist

- **Missing packages?** Re-run `make setup` or `pip install -r requirements.txt`.
- **Python version errors?** The project requires Python 3.11 or newer. Check your version with `python --version`.
- **Model file not found?** Train the model first with `make train` or `python src/train_model.py`, or provide the correct
  path to `--model`.
- **Makefile commands don't work?** On Windows, you may need to install Make or use the manual Python commands instead.
  Alternatively, use Git Bash or Windows Subsystem for Linux (WSL).
- **File upload fails in Streamlit?** Ensure the CSV has column names that match the dataset. For JSON, wrap multiple employees
  in a list (`[{...}, {...}]`).
- **Predictions look unrealistic?** Confirm that dates use the format `YYYY-MM-DD`, and that salaries and survey scores are in
  the same ranges as the training data (see `docs/DATA_CARD.md` for expected ranges).
- **Linting or formatting errors?** Run `make format` to auto-fix code style issues, or `make lint` to just check.
- **Tests are failing?** Run `make test` to see detailed test results with coverage reports. Tests are stored in the `tests/`
  folder.
- **Want to start fresh?** Run `make clean` to remove all generated files (models, reports, caches) and start over.

---

## 11. Glossary

### Machine Learning Terms

- **Feature:** A column in the dataset describing an employee (e.g., salary, tenure, department).
- **Label/Target:** The value you want to predict (here: `Termd`, indicating whether someone was terminated).
- **Pipeline:** A bundle of preprocessing steps and a model executed in sequence. Ensures consistency between training and
  prediction.
- **Hyperparameters:** Settings chosen before training (e.g., how many trees in a forest, learning rate). Found through grid
  search over candidate values.
- **Cross-validation:** Splitting the training data into folds to evaluate models more reliably. The project uses 5-fold
  stratified cross-validation.
- **Overfitting:** When a model memorizes training data and performs poorly on new data. Techniques like cross-validation,
  stratified splitting, and random forests reduce this risk.

### Statistical Concepts

- **Bootstrap:** A resampling technique where you repeatedly draw random samples (with replacement) from your data to estimate
  confidence intervals. The project uses 1,000 bootstrap samples for test metrics.
- **Confidence Interval (CI):** A range of values that likely contains the true metric. "95% CI" means if you repeated the
  experiment 100 times, about 95 would have confidence intervals containing the true value.
- **Stratification:** Maintaining the same class distribution when splitting data. Ensures the training, validation, and test
  sets have similar proportions of terminated vs. active employees.
- **Class Imbalance:** When one outcome is much more common than another. Here, there are roughly twice as many active employees
  as terminated ones. Addressed with RandomOverSampler.

### Model Evaluation Terms

- **False Positive (Type I Error):** Predicting termination when the employee actually stays. Can lead to unnecessary
  interventions.
- **False Negative (Type II Error):** Predicting retention when the employee actually leaves. Means missing opportunities to
  intervene.
- **Baseline Model:** A simple comparison point. If your complex model doesn't beat a baseline (like always predicting the most
  common class), it's not useful.
- **Ablation Study:** An experiment where you remove components to see how much they contribute. Helps understand what's
  essential vs. optional.

### Project-Specific Terms

- **Permutation Importance:** A method for measuring feature importance by randomly shuffling each feature and seeing how much
  model performance drops.
- **Calibration:** How well predicted probabilities match reality. A well-calibrated model that says "30% chance" should be
  correct about 30% of the time.
- **Makefile Targets:** Commands defined in the `Makefile` that automate common tasks (`make train`, `make test`, etc.).
- **Pre-commit Hooks:** Automated checks that run before you save code changes. Ensure code style consistency (black, isort,
  flake8).

### Reproducibility Terms

- **Random Seed:** A number that initializes random number generators. Using the same seed (here: 42) ensures experiments
  produce identical results every time.
- **Pinned Dependencies:** Exact version numbers for all software packages. Prevents "works on my machine" problems.
- **CI/CD (Continuous Integration/Continuous Deployment):** Automated workflows that test and deploy code. The project uses
  GitHub Actions (`.github/workflows/ci.yml` and `.github/workflows/retrain-model.yml`) to automatically run tests and retrain
  models.

---

## 12. Summary and next steps

You are now equipped to run the project end-to-end and interpret every output. Here's a quick reference for common workflows:

### Quick Start Checklist

1. **Setup:** Run `make reproduce` for a full automated pipeline, or `make setup` + `make train` for manual control
2. **Review outputs:** Check `reports/metrics_with_ci.md` for model performance with confidence intervals
3. **Examine diagnostics:** Open the PNG plots in `reports/figures/` to visualize model behavior
4. **Understand failures:** Read `reports/FAILURES.md` to see where the model struggles
5. **Make predictions:** Use `make app` for the GUI or `python src/predict_cli.py` for the command line
6. **Take action:** Consult `docs/BUSINESS_ACTIONS.md` to translate risk scores into HR interventions

### Learning Resources

- **For dataset details:** Read `docs/DATA_CARD.md`
- **For technical overview:** See the main `README.md`
- **For code quality:** Review `.pre-commit-config.yaml` to understand automated checks
- **For automation:** Examine `.github/workflows/` to see CI/CD pipelines

### Getting Help

- **Documentation bugs or unclear sections?** Open an issue on the project's GitHub repository
- **Want to customize for your organization?** Start by editing `configs/policy.yaml` and retraining with your data
- **Need to understand the code?** The codebase follows standard Python conventions with comprehensive docstrings

Keep this guide nearby as a reference whenever questions come up. Remember: predictions are tools to support conversations, not
replace human judgment in HR decisions.
