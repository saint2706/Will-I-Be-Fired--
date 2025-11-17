# Experiment Configuration Guide

Experiment YAML files live under `configs/experiments/` and describe every aspect of a training run. Each file mirrors the structure consumed by `src/train_model.py --config ...`.

## File Structure

```yaml
data:
  dataset_path: ../../HRDataset_v14.csv

preprocessing:
  include_numeric_features:
    - Salary
  exclude_numeric_features:
    - Absences
  include_categorical_features:
    - Department
  exclude_categorical_features:
    - State

models:
  custom_rf:
    type: random_forest
    model_parameters:
      class_weight: balanced
    hyperparameters:
      model__n_estimators: [100, 200]
      model__max_depth: [8, 12]

cross_validation:
  n_splits: 5
  shuffle: true
  random_state: 42
```

### Sections
- **data** – absolute or relative path to the CSV dataset. Relative paths resolve from the config file's directory.
- **preprocessing** – optional lists to explicitly include or exclude numeric/categorical features. Omitted fields fall back to the defaults in `src/feature_engineering.py`.
- **models** – dictionary of model definitions. Each key becomes the model name in metrics. Provide:
  - `type`: one of `logistic_regression`, `random_forest`, or `gradient_boosting`.
  - `model_parameters` *(optional)*: base estimator keyword arguments (e.g., `class_weight`).
  - `hyperparameters`: parameter grid passed directly to `GridSearchCV`.
- **cross_validation** – values forwarded to `StratifiedKFold`.

## Tips
- Start from `experiments/lightweight.yaml` for quick smoke tests.
- Duplicate `experiments/default.yaml` for full reproductions.
- Keep grids small when experimenting to avoid long runtimes.
- Store custom experiment files alongside the provided templates to simplify version control.
