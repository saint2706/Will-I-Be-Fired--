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

## 2. How the data becomes predictions

The workflow follows six repeatable stages. You can re-run them with the scripts in the `src/` folder.

1. **Load the data** – Read the CSV file into memory using pandas (a spreadsheet-like Python library). Every column is kept in
a structured format so later steps can understand whether a value is text, a number, or a date.
2. **Clean and enrich the features** – Convert human-readable dates into machine-friendly numbers (for example tenure in years
   or how long since the last review), drop columns that could leak the true outcome (for example the literal termination date),
   fill in missing values, and convert words into numbers so algorithms can work with them. The code that does this lives in
   [`src/feature_engineering.py`](../src/feature_engineering.py).
3. **Split the dataset** – Divide the rows into three non-overlapping pieces:
   - **Training set (60%)** – What the algorithms learn from.
   - **Validation set (20%)** – Used during tuning to choose the best algorithm without touching the final test.
   - **Test set (20%)** – Held back until the end so we know how the chosen model behaves on truly unseen people.
   The splitting keeps the same proportion of terminated vs. active employees in each subset (this is called *stratification*).
4. **Train multiple models** – Three popular machine-learning approaches are tried:
   - **Logistic regression** – A simple formula that estimates the log-odds of termination as a weighted sum of the input
     features.
   - **Random forest** – A collection of decision trees that vote together. Each tree looks at random subsets of features to
     avoid overfitting.
   - **Gradient boosting** – Builds trees one at a time, each focusing on correcting the mistakes of the previous ones.
5. **Evaluate and select the winner** – Every model is assessed with accuracy, precision, recall, and ROC-AUC (explained in
   Section 4). The model with the best validation ROC-AUC is kept, then tested one final time on the test set to estimate
   real-world performance.
6. **Save the pipeline** – The winning model and every preprocessing step are bundled together into a single pipeline object and
   stored as `models/best_model.joblib`. When you load it later, it remembers how to clean new records in the same way.

The entire routine is automated in [`src/train_model.py`](../src/train_model.py). Running that script from the command line
recreates every experiment and produces a metrics report in `reports/metrics.json` for transparency.

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

Every metric is calculated twice: once on the validation set and once on the test set. High scores across both are a good sign.

- **Accuracy** – The share of all predictions the model gets right. Formula: `(true positives + true negatives) / all cases`.
  Works best when both classes are balanced; for imbalanced data it can be misleading.
- **Precision** – When the model predicts someone will be terminated, precision measures how often that is actually true.
  Formula: `true positives / (true positives + false positives)`.
- **Recall (also called sensitivity)** – Out of everyone who really gets terminated, recall tells you how many the model caught.
  Formula: `true positives / (true positives + false negatives)`.
- **ROC-AUC** – Stands for *Receiver Operating Characteristic – Area Under the Curve*. The model produces a probability between
  0 and 1. ROC-AUC looks at every possible probability threshold and measures how well the model separates terminated from active
  employees. A perfect score is 1.0; 0.5 means guessing.
- **Confidence score in predictions** – When you request future risks at specific tenure milestones (1, 2, 5 years), the tool
  reports both the predicted probability and a confidence value. The confidence is derived from the model's probability output
  by measuring how far it sits from uncertainty (0.5). The further away, the more confident the model is about that prediction.

---

## 5. Step-by-step: running the project without writing code

Follow these steps on macOS, Windows (with PowerShell), or Linux. Replace `<path>` with the actual folder where you stored the
repository.

1. **Install Python 3.10 or newer.**
2. **Open a terminal** (Command Prompt, PowerShell, or Terminal app).
3. **Move into the project folder:**

   ```bash
   cd <path>/Will-I-Be-Fired--
   ```

4. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Train or retrain the model (optional):**

   ```bash
   python src/train_model.py
   ```

   This command prints progress logs, saves the best model to `models/best_model.joblib`, and writes detailed metrics to
   `reports/metrics.json`.
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

### 6.2 Metrics report (`reports/metrics.json`)

This file is a JSON object with:

- `validation` and `test` sections, each containing accuracy, precision, recall, and ROC-AUC.
- The name of the chosen model.
- The hyperparameters used during training.
Open it in any text editor or online JSON viewer to explore the numbers.

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

---

## 7. Ethics, responsible use, and monitoring

- **Bias checks:** Periodically review performance separately for different demographic groups.
- **Explainability:** Use the feature importances from the random forest or shapley value tools to explain why a prediction was
  high or low.
- **Data privacy:** Keep the dataset and prediction reports in secure storage with strict access controls.
- **Retraining cadence:** Schedule a review (for example every quarter). If the input data distribution changes or if metrics on
  fresh data degrade, re-run `train_model.py` to produce an updated model.

---

## 8. Troubleshooting checklist

- **Missing packages?** Re-run `pip install -r requirements.txt`.
- **Model file not found?** Train the model first (`python src/train_model.py`) or provide the correct path to `--model`.
- **File upload fails in Streamlit?** Ensure the CSV has column names that match the dataset. For JSON, wrap multiple employees
  in a list (`[{...}, {...}]`).
- **Predictions look unrealistic?** Confirm that dates use the format `YYYY-MM-DD`, and that salaries and survey scores are in
  the same ranges as the training data.

---

## 9. Glossary

- **Feature:** A column in the dataset describing an employee.
- **Label/Target:** The value you want to predict (here: `Termd`).
- **Pipeline:** A bundle of preprocessing steps and a model executed in sequence.
- **Hyperparameters:** Settings chosen before training (e.g. how many trees in a forest).
- **Cross-validation:** Splitting the training data into folds to evaluate models more reliably.
- **Overfitting:** When a model memorises training data and performs poorly on new data. Techniques like cross-validation and
  random forests reduce this risk.

You are now equipped to run the project end-to-end and interpret every output. Keep this guide nearby as a reference whenever
questions come up.
