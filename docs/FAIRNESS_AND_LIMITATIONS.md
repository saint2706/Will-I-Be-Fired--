# Fairness & Responsible Use Report

This note summarises the bias/fairness audit that now runs as part of the
training pipeline (`make train` automatically calls `python src/fairness_analysis.py`).
The script reloads the persisted best model (`models/best_model.joblib`),
recreates the canonical 70/15/15 train/validation/test split, and evaluates the
held-out test set for the protected attributes available in the dataset:

- `RaceDesc`
- `Sex`
- `MaritalDesc`
- `CitizenDesc`

For every group we compute accuracy, precision, recall, ROC-AUC, Brier score
(calibration proxy), positive-prediction rate, demographic parity difference
(max minus min positive-prediction rate within the attribute), and equal
opportunity difference (max minus min recall for the positive class). The tidy
output is saved to `reports/fairness_summary.csv` and mirrored in the bar charts
stored under `reports/figures/fairness_<attribute>.png`.

## Key findings (current dataset + model)

| Attribute | Largest demographic parity gap | Largest equal opportunity gap | Notes |
|-----------|--------------------------------|-------------------------------|-------|
| RaceDesc | **0.39** between White (39% positive predictions) and Two or more races/American Indian (0%). | **1.00** because some groups have zero actual terminations in the test split, so recall collapses to 0. | Group counts range from 1–23 employees; treat these stats as directional only. |
| Sex | 0.10 between men (29% positive predictions) and women (39%). | 0.00 because both groups achieved 100% recall on the tiny test slice. | Apparent parity is driven by the near-perfect test accuracy, not necessarily real-world equality. |
| MaritalDesc | **0.57** between divorced employees (57% predicted positive) and separated employees (0%). | **1.00** for the same reason: some categories have no actual terminations. | Very small cells (e.g., only 2 separated or widowed employees). |
| CitizenDesc | 0.01 between citizens and eligible non-citizens. | 0.00 with the current test set. | Counts are still small (3 non-citizens). |

Interpretation tips:

- Perfect accuracy/recall numbers reflect the tiny test set (47 employees) and
  oversampled training regime; they **do not** guarantee the absence of bias.
- Demographic parity gaps above ~0.2 already warrant investigation, but here the
  largest gaps coincide with groups that contain ≤2 people. Statistical
  confidence is therefore extremely low.

## Limitations & warnings

1. **Sample size** – The test split contains only 47 employees. Several protected
   groups have 1–3 members, so any gap can swing wildly with a single record.
2. **Historical bias** – The raw HR dataset may encode biased termination
   decisions. The model learns from that history; even perfect parity on this
   dataset would not prove fairness in practice.
3. **Feature coverage** – We only audited attributes present in the dataset.
   Other sensitive traits (disability status, parental leave history, etc.) are
   missing entirely and therefore unaudited.
4. **Model confidence** – Reported recall/precision of 1.0 stems from
   overfitting on a small sample. Real-world deployment will introduce errors
   that may not be evenly distributed across groups.
5. **Responsible use** – The model must **never** be used as an automated firing
   engine. It is designed to surface retention risks for **human review**, not
   to trigger adverse employment actions without context.

## How to reproduce

Run the fairness module directly or via the Makefile once the best model exists:

```bash
# Stand-alone run
python src/fairness_analysis.py

# Part of the full pipeline
make train   # trains + runs fairness analysis automatically
make fairness  # re-run audit using the persisted model
```

Inspect the generated CSV/figures alongside this document before sharing
predictions with stakeholders.
