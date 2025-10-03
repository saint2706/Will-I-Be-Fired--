# AI Coding Agent Instructions for "Will I Be Fired?" Project

## Project Architecture Overview
This is an **employee termination prediction system** built with scikit-learn. The core architecture consists of:

- **Feature Engineering Pipeline** (`src/feature_engineering.py`) - Centralized data transformations for temporal features, column cleanup, and schema definitions
- **Training System** (`src/train_model.py`) - Cross-validated model selection comparing LogisticRegression, RandomForest, and GradientBoosting
- **Inference Layer** (`src/inference.py`) - Reusable prediction functions with tenure-based risk analysis
- **Multiple UIs** - CLI (`predict_cli.py`), Streamlit GUI (`gui_app.py`) for different user workflows
- **Centralized Logging** (`logging_utils.py`) - Consistent logging format across all components

## Key Data Flow Patterns

### Feature Schema Consistency
All components use **centralized feature lists** from `feature_engineering.py`:
```python
NUMERIC_FEATURES = ("Salary", "EngagementSurvey", ..., "tenure_years", "age_years", "years_since_last_review")
CATEGORICAL_FEATURES = ("Department", "PerformanceScore", ..., "HispanicLatino")
DROP_COLUMNS = ("Employee_Name", "EmpID", ..., "TermReason")  # Identifiers and leakage sources
```

### Temporal Feature Engineering
Key engineered features are calculated from date columns:
- `tenure_years` from `DateofHire`
- `age_years` from `DOB`
- `years_since_last_review` from `LastPerformanceReview_Date`

**Critical**: Use `prepare_training_data()` for training, `prepare_inference_frame()` for predictions to ensure consistent preprocessing.

### Model Pipeline Structure
Uses imbalanced-learn's `ImbPipeline` with RandomOverSampler:
```python
Pipeline([
    ('preprocessor', ColumnTransformer with scaling/encoding),
    ('sampler', RandomOverSampler),
    ('classifier', model)
])
```

## Development Workflows

### Training Models
```bash
python src/train_model.py  # Trains all models, saves best to models/best_model.joblib
```
- Uses 60/20/20 stratified train/validation/test split
- Grid search with 5-fold cross-validation
- Selects best model by validation ROC-AUC
- Outputs comprehensive metrics to `reports/metrics.json`

### Running Inference
```bash
# CLI interface
python src/predict_cli.py --model models/best_model.joblib

# GUI interface  
streamlit run src/gui_app.py
```

### Installation & Dependencies
```bash
pip install -r requirements.txt  # Core: pandas, scikit-learn, imbalanced-learn, streamlit
```

## Project-Specific Conventions

### Import Patterns
All modules use **relative imports with fallback** for both package and script execution:
```python
try:
    from .feature_engineering import prepare_training_data
except ImportError:  # pragma: no cover - fallback for script execution
    from feature_engineering import prepare_training_data
```

### Error Handling & Logging
- Initialize logging with `configure_logging()` from `logging_utils.py`
- Use `get_logger(__name__)` in each module
- Robust date parsing with fallback logic in feature engineering
- All scripts handle missing model files gracefully

### File Structure Conventions
- Models saved to `models/` (gitignored, regenerate with training script)
- Metrics reports in `reports/metrics.json` 
- Source code modularized in `src/` with clear separation of concerns
- Documentation includes both technical README and `docs/non_programmer_guide.md`

### Tenure Risk Analysis
Special inference pattern for **multi-horizon predictions**:
```python
from inference import predict_tenure_risk
risks = predict_tenure_risk(employee_record, horizons=[1, 2, 5])  # Years
# Returns TenureRisk objects with probability + confidence
```

## Target Dataset Characteristics
- **Small dataset**: 311 employees (104 terminated, 207 active)
- **Imbalanced**: ~1:2 ratio handled with RandomOverSampler in pipeline
- **High performance**: Best models achieve perfect test metrics (potential overfitting indicator)
- **Sensitive data**: Handle employee records with appropriate privacy considerations

## Model Persistence & Loading
Models are saved as complete scikit-learn pipelines with preprocessing:
```python
model = joblib.load("models/best_model.joblib")
# Can directly predict on raw employee dicts/DataFrames
```

When modifying preprocessing logic, always retrain and re-save the complete pipeline to maintain consistency between training and inference.