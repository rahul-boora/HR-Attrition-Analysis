# HR Employee Attrition Analysis

End-to-end analysis of employee attrition: exploratory analysis in Python, a predictive Random Forest model, and an interactive Power BI dashboard that scores every employee with an attrition risk band.

## Problem Statement

Attrition costs companies 50–200% of an employee's annual salary in replacement and ramp-up costs. The goal of this project is to (1) identify the key drivers of attrition and (2) flag at-risk employees before they leave, so HR can intervene.

## Dataset

IBM HR Analytics Employee Attrition & Performance (Kaggle) — 1,470 employees, 35 features covering demographics, compensation, job role, satisfaction scores, and tenure. Target variable: `Attrition` (Yes/No, ~16% positive class).

Download: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

## Tech Stack

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn), Power BI (DAX, decomposition tree, conditional formatting).

## Approach

1. **Cleaning** — dropped zero-variance columns (`EmployeeCount`, `StandardHours`, `Over18`), verified no missing values, encoded the target.
2. **EDA** — attrition rates across categorical features, KDE distributions of leavers vs stayers, correlation heatmap.
3. **Encoding** — label encoding for binary features, one-hot encoding for multi-class features.
4. **Modelling** — stratified 80/20 split; Logistic Regression baseline and Random Forest, both with `class_weight="balanced"` to handle the 16/84 class imbalance.
5. **Evaluation** — classification report, confusion matrix, ROC-AUC (~0.80+ for Random Forest).
6. **Power BI** — exported a scored dataset with `AttritionRiskScore` and `RiskBand` (Low/Medium/High) and built a 3-page dashboard: Overview KPIs, Drivers, and a High-Risk Watchlist.

## Key Insights

- Overtime is the single strongest driver: ~31% attrition among overtime workers vs ~10% otherwise.
- Sales Representatives and Laboratory Technicians have the highest role-level attrition.
- Risk concentrates in the first 2 years of tenure and in the lowest monthly-income bands.
- Single employees leave at roughly double the rate of married employees.

## Business Recommendations

- Cap or compensate sustained overtime, especially in Sales and Lab Technician roles.
- Strengthen onboarding and 6/12/18-month check-ins for early-tenure employees.
- Review compensation bands at the lower end against market benchmarks.
- Use the High-Risk Watchlist for proactive manager 1:1s.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
# place WA_Fn-UseC_-HR-Employee-Attrition.csv in this folder
python hr_attrition_analysis.py
```

Outputs: EDA charts (PNG), model evaluation charts, and `hr_attrition_powerbi.csv` for the Power BI dashboard.
