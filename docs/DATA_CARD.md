# Data Card: HRDataset_v14.csv

## Dataset Provenance

- **Source:** `HRDataset_v14.csv`
- **Size:** 311 employee records
- **Target Variable:** `Termd` (1 = terminated, 0 = active)
- **Class Distribution:** 104 terminated (33.4%), 207 active (66.6%)
- **Collection Period:** Historical HR records with dates ranging from early 2000s to 2020s
- **Purpose:** Train machine learning models to predict employee termination risk

## Feature Schema

### Numeric Features

| Feature | Type | Description | Missing % | Notes |
|---------|------|-------------|-----------|-------|
| `Salary` | int | Annual salary in USD | 0% | Range: ~$45k-$250k |
| `EngagementSurvey` | float | Employee engagement score (1-5) | ~5% | Survey-based metric |
| `EmpSatisfaction` | int | Employee satisfaction score (1-5) | ~3% | Self-reported |
| `SpecialProjectsCount` | int | Number of special projects | 0% | Range: 0-6 |
| `DaysLateLast30` | int | Days late in last 30 days | 0% | Attendance metric |
| `Absences` | int | Number of absences | 0% | Attendance metric |

### Engineered Temporal Features

| Feature | Type | Description | Missing % | Derivation |
|---------|------|-------------|-----------|------------|
| `tenure_years` | float | Years since hire date | 0% | Computed from `DateofHire` |
| `age_years` | float | Employee age in years | 0% | Computed from `DOB` |
| `years_since_last_review` | float | Years since last performance review | ~2% | Computed from `LastPerformanceReview_Date` |

### Categorical Features

| Feature | Type | Description | Cardinality | Missing % |
|---------|------|-------------|-------------|-----------|
| `Department` | str | Department name | 6 | 0% |
| `PerformanceScore` | str | Performance rating | 4 | 0% |
| `RecruitmentSource` | str | How employee was recruited | 8 | 0% |
| `Position` | str | Job title | 29 | 0% |
| `State` | str | US state of employment | 12 | 0% |
| `Sex` | str | Gender (Male/Female) | 2 | 0% |
| `MaritalDesc` | str | Marital status | 4 | 0% |
| `CitizenDesc` | str | Citizenship status | 3 | 0% |
| `RaceDesc` | str | Race/ethnicity | 5 | 0% |
| `HispanicLatino` | str | Hispanic/Latino (Yes/No) | 2 | 0% |

### Date Columns (Used for Feature Engineering Only)

| Column | Type | Description | Missing % | Usage |
|--------|------|-------------|-----------|-------|
| `DateofHire` | date | Employee hire date | 0% | Used to compute tenure |
| `DOB` | date | Date of birth | 0% | Used to compute age |
| `LastPerformanceReview_Date` | date | Last performance review date | ~2% | Used to compute review recency |
| `DateofTermination` | date | Termination date (if applicable) | 66%* | Used to compute tenure for terminated employees |

*Note: Missing for active employees (expected)

## Dropped Columns (Leakage & Identifiers)

The following columns are **removed before modeling** to prevent data leakage and protect privacy:

### Identifier Columns (Dropped)
- `Employee_Name` - Personal identifier
- `EmpID` - Unique employee ID
- `MarriedID`, `MaritalStatusID`, `GenderID`, `EmpStatusID`, `DeptID`, `PerfScoreID`, `FromDiversityJobFairID` - Redundant numeric IDs

### Leakage-Prone Columns (Dropped)
- `ManagerName` - Could encode termination patterns specific to managers not generalizable
- `ManagerID` - Same concern as ManagerName
- `EmploymentStatus` - **Directly leaks target** (e.g., "Terminated" status)
- `TermReason` - **Directly leaks target** (only populated for terminated employees)

### Date Columns (Dropped After Feature Engineering)
- `DateofTermination` - Used for tenure calculation, then dropped
- All other date columns - Converted to numeric features, then dropped

## Data Quality Notes

1. **Missing Values:** Generally low (<5%) for most features. The preprocessing pipeline handles missing values via:
   - Numeric features: Median imputation
   - Categorical features: Most-frequent imputation

2. **Temporal Consistency:** Dates are parsed with robust fallback logic to handle multiple formats.

3. **Class Imbalance:** The 1:2 imbalance (terminated:active) is addressed via RandomOverSampler in the training pipeline.

## Potential Biases & Fairness Considerations

- **Historical Bias:** The dataset reflects historical HR decisions, which may encode biases in hiring, promotion, or termination.
- **Protected Attributes:** Features like `Sex`, `RaceDesc`, and `HispanicLatino` are included but should be monitored for disparate impact.
- **Manager Effects:** Manager-specific termination patterns are intentionally excluded to improve generalization.

## Recommended Monitoring

- Track feature distributions over time for drift detection
- Monitor model performance across demographic subgroups
- Audit false positive/negative rates by protected attributes
- Regularly retrain with fresh data to adapt to changing HR practices
