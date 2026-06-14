"""
HR Employee Attrition Analysis
================================
Dataset : IBM HR Analytics Employee Attrition & Performance (Kaggle)
          File: WA_Fn-UseC_-HR-Employee-Attrition.csv (1,470 rows, 35 columns)
Goal    : Identify the key drivers of employee attrition and build a
          predictive model to flag at-risk employees.
Stack   : Python (Pandas, Matplotlib, Seaborn, Scikit-learn) + Power BI dashboard
Author  : Rahul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(f"Shape: {df.shape}")
print(df["Attrition"].value_counts(normalize=True).round(3))   # ~16% attrition

# ----------------------------------------------------------------------------
# 2. DATA CLEANING
# ----------------------------------------------------------------------------
# Drop columns with zero variance / no analytical value
df = df.drop(columns=["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"])

# No missing values in this dataset, but always verify
assert df.isnull().sum().sum() == 0, "Missing values found!"

# Target encoding: Yes -> 1, No -> 0
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# ----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------------
# 3.1 Attrition rate by key categorical features
cat_features = ["Department", "JobRole", "OverTime", "MaritalStatus",
                "BusinessTravel", "Gender", "EducationField"]

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()
for i, col in enumerate(cat_features):
    rate = df.groupby(col)["Attrition"].mean().sort_values(ascending=False)
    sns.barplot(x=rate.values, y=rate.index, ax=axes[i], palette="rocket")
    axes[i].set_title(f"Attrition Rate by {col}")
    axes[i].set_xlabel("Attrition Rate")
fig.delaxes(axes[-1])
plt.tight_layout()
plt.savefig("eda_categorical_attrition.png", bbox_inches="tight")
plt.close()

# 3.2 Numeric distributions: leavers vs stayers
num_features = ["Age", "MonthlyIncome", "DistanceFromHome",
                "YearsAtCompany", "TotalWorkingYears", "JobSatisfaction"]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(num_features):
    sns.kdeplot(data=df, x=col, hue="Attrition", fill=True,
                common_norm=False, ax=axes[i], palette=["#2a9d8f", "#e76f51"])
    axes[i].set_title(f"{col}: Stayers (0) vs Leavers (1)")
plt.tight_layout()
plt.savefig("eda_numeric_distributions.png", bbox_inches="tight")
plt.close()

# 3.3 Correlation heatmap (numeric features)
plt.figure(figsize=(16, 12))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Correlation Heatmap — Numeric Features")
plt.savefig("eda_correlation_heatmap.png", bbox_inches="tight")
plt.close()

# Key EDA insights (printed for report):
print("""
KEY EDA INSIGHTS
----------------
1. OverTime employees attrite at ~31% vs ~10% for non-overtime — strongest single driver.
2. Sales Representatives and Lab Technicians show the highest role-level attrition.
3. Younger employees (< 30) and low MonthlyIncome bands leave significantly more.
4. Attrition is highest in the first 2 years at the company (early-tenure risk).
5. Single employees attrite at roughly double the rate of married/divorced.
""")

# ----------------------------------------------------------------------------
# 4. FEATURE ENGINEERING & ENCODING
# ----------------------------------------------------------------------------
df_model = df.copy()

# Label-encode binary columns, one-hot encode multi-class columns
binary_cols = [c for c in df_model.select_dtypes("object").columns
               if df_model[c].nunique() == 2]
multi_cols  = [c for c in df_model.select_dtypes("object").columns
               if df_model[c].nunique() > 2]

le = LabelEncoder()
for col in binary_cols:
    df_model[col] = le.fit_transform(df_model[col])

df_model = pd.get_dummies(df_model, columns=multi_cols, drop_first=True)

X = df_model.drop(columns="Attrition")
y = df_model["Attrition"]

# Stratified split — preserves the 16/84 class ratio in train & test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features (needed for Logistic Regression)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ----------------------------------------------------------------------------
# 5. MODELLING
# ----------------------------------------------------------------------------
# 5.1 Baseline: Logistic Regression with class_weight to handle imbalance
log_reg = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=42)
log_reg.fit(X_train_sc, y_train)
y_pred_lr = log_reg.predict(X_test_sc)
print("\n--- Logistic Regression ---")
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", round(roc_auc_score(y_test,
      log_reg.predict_proba(X_test_sc)[:, 1]), 3))

# 5.2 Random Forest with balanced class weights
rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                            class_weight="balanced", random_state=42,
                            n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\n--- Random Forest ---")
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", round(roc_auc_score(y_test,
      rf.predict_proba(X_test)[:, 1]), 3))

# Confusion matrix for the better model
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d",
            cmap="Blues", cbar=False,
            xticklabels=["Stay", "Leave"], yticklabels=["Stay", "Leave"])
plt.title("Random Forest — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.savefig("model_confusion_matrix.png", bbox_inches="tight")
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_estimator(log_reg, X_test_sc, y_test,
                               name="Logistic Regression", ax=ax)
RocCurveDisplay.from_estimator(rf, X_test, y_test,
                               name="Random Forest", ax=ax)
plt.title("ROC Curve Comparison")
plt.savefig("model_roc_curves.png", bbox_inches="tight")
plt.close()

# ----------------------------------------------------------------------------
# 6. FEATURE IMPORTANCE — what actually drives attrition?
# ----------------------------------------------------------------------------
importances = (pd.Series(rf.feature_importances_, index=X.columns)
               .sort_values(ascending=False).head(15))
plt.figure(figsize=(10, 7))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Top 15 Drivers of Attrition (Random Forest)")
plt.xlabel("Feature Importance")
plt.savefig("feature_importance.png", bbox_inches="tight")
plt.close()
print("\nTop 10 attrition drivers:\n", importances.head(10))

# ----------------------------------------------------------------------------
# 7. EXPORT FOR POWER BI
# ----------------------------------------------------------------------------
# Score every employee with attrition risk probability for the dashboard
df_export = df.copy()
df_export["AttritionRiskScore"] = rf.predict_proba(X)[:, 1].round(3)
df_export["RiskBand"] = pd.cut(df_export["AttritionRiskScore"],
                               bins=[0, 0.3, 0.6, 1.0],
                               labels=["Low", "Medium", "High"])
df_export.to_csv("hr_attrition_powerbi.csv", index=False)
print("\nExported hr_attrition_powerbi.csv for Power BI dashboard.")

"""
POWER BI DASHBOARD (built on hr_attrition_powerbi.csv)
------------------------------------------------------
Pages:
1. Overview      — KPI cards (Headcount, Attrition Rate %, Avg Tenure,
                   Avg Monthly Income), attrition trend by tenure bucket.
2. Drivers       — attrition rate by OverTime, JobRole, Department,
                   income band (decomposition tree visual).
3. Risk Watchlist— table of High-risk employees with slicers on
                   Department / JobRole, conditional formatting on RiskBand.

Key DAX measures:
  Attrition Rate = DIVIDE(CALCULATE(COUNTROWS(HR), HR[Attrition]=1),
                          COUNTROWS(HR))
  High Risk Count = CALCULATE(COUNTROWS(HR), HR[RiskBand]="High")
"""
