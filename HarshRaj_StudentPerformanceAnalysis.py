"""
Student Performance Analysis
Author: Harsh Raj
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

Research Question:
What social, economic and educational factors are associated with differences
in student academic performance?
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 1. LOAD DATA
# ============================================================

# The script supports both the GitHub project structure and a
# dataset placed in the same folder as this Python file.
possible_paths = [
    os.path.join("data", "StudentPerformanceFactors.csv"),
    "StudentPerformanceFactors.csv"
]

file_path = next(
    (path for path in possible_paths if os.path.exists(path)),
    None
)

if file_path is None:
    raise FileNotFoundError(
        "StudentPerformanceFactors.csv was not found. "
        "Place it in the data/ folder or the same folder as this script."
    )

df = pd.read_csv(file_path)

os.makedirs("outputs", exist_ok=True)

print("\n========== DATASET OVERVIEW ==========")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 2. DESCRIPTIVE SUMMARY
# ============================================================

print("\n========== NUMERICAL SUMMARY ==========")
print(df.describe())

print("\n========== CATEGORICAL SUMMARY ==========")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False))


# ============================================================
# 3. DATA CLEANING
# ============================================================

clean_df = df.copy()

# These variables are categorical, so missing values are
# replaced with their respective modal category.
categorical_missing = [
    "Teacher_Quality",
    "Parental_Education_Level",
    "Distance_from_Home"
]

for column in categorical_missing:
    clean_df[column] = clean_df[column].fillna(
        clean_df[column].mode()[0]
    )

print("\n========== CLEANING RESULTS ==========")
print("Rows after cleaning:", len(clean_df))
print("Columns after cleaning:", len(clean_df.columns))
print("Remaining missing values:", clean_df.isnull().sum().sum())
print("Remaining duplicate rows:", clean_df.duplicated().sum())


# ============================================================
# 4. OUTLIER / DATA-QUALITY CHECK
# ============================================================

score_101_count = (clean_df["Exam_Score"] == 101).sum()

mean_with_101 = clean_df["Exam_Score"].mean()

clean_without_101 = clean_df[
    clean_df["Exam_Score"] <= 100
].copy()

mean_without_101 = clean_without_101["Exam_Score"].mean()

print("\n========== EXAM SCORE DATA-QUALITY CHECK ==========")
print("Exam Score values above 100:", score_101_count)
print("Mean Exam Score with 101:", round(mean_with_101, 3))
print("Mean Exam Score without 101:", round(mean_without_101, 3))
print(
    "Difference in mean:",
    round(abs(mean_with_101 - mean_without_101), 3)
)


# ============================================================
# 5. EXPLORATORY ANALYSIS
# ============================================================

# ----- Attendance -----

attendance_correlation = clean_df[
    "Attendance"
].corr(clean_df["Exam_Score"])

clean_df["Attendance_Group"] = pd.cut(
    clean_df["Attendance"],
    bins=[59, 69, 79, 89, 100],
    labels=["60-69", "70-79", "80-89", "90-100"]
)

attendance_summary = clean_df.groupby(
    "Attendance_Group",
    observed=True
)["Exam_Score"].agg(["count", "mean"]).round(2)

print("\n========== ATTENDANCE ANALYSIS ==========")
print("Correlation:", round(attendance_correlation, 3))
print(attendance_summary)


# ----- Study Hours -----

study_correlation = clean_df[
    "Hours_Studied"
].corr(clean_df["Exam_Score"])

clean_df["Study_Hours_Group"] = pd.cut(
    clean_df["Hours_Studied"],
    bins=[0, 9, 19, 29, 44],
    labels=["1-9", "10-19", "20-29", "30-44"]
)

study_summary = clean_df.groupby(
    "Study_Hours_Group",
    observed=True
)["Exam_Score"].agg(["count", "mean"]).round(2)

print("\n========== STUDY HOURS ANALYSIS ==========")
print("Correlation:", round(study_correlation, 3))
print(study_summary)


# ----- Access to Resources -----

resource_summary = (
    clean_df.groupby("Access_to_Resources")["Exam_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)

print("\n========== ACCESS TO RESOURCES ==========")
print(resource_summary)


# ----- Parental Involvement -----

involvement_summary = (
    clean_df.groupby("Parental_Involvement")["Exam_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)

print("\n========== PARENTAL INVOLVEMENT ==========")
print(involvement_summary)


# ----- Family Income -----

income_summary = (
    clean_df.groupby("Family_Income")["Exam_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)

print("\n========== FAMILY INCOME ==========")
print(income_summary)


# ----- Parental Education -----

education_summary = (
    clean_df.groupby("Parental_Education_Level")["Exam_Score"]
    .agg(["count", "mean", "std"])
    .round(2)
)

print("\n========== PARENTAL EDUCATION ==========")
print(education_summary)


# ============================================================
# 6. MULTIPLE LINEAR REGRESSION — SCIKIT-LEARN
# ============================================================

numerical_features = [
    "Attendance",
    "Hours_Studied",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity"
]

categorical_features = [
    "Family_Income",
    "Parental_Education_Level",
    "Access_to_Resources",
    "Parental_Involvement",
    "Motivation_Level"
]

X = clean_df[
    numerical_features + categorical_features
]

y = clean_df["Exam_Score"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                drop="first",
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)

model.fit(X, y)

# In-sample predictions are used here for descriptive model fit.
y_pred = model.predict(X)

r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print("\n========== MULTIPLE LINEAR REGRESSION ==========")
print("R-squared:", round(r2, 3))
print("Mean Absolute Error:", round(mae, 3))

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = model.named_steps[
    "regression"
].coef_

coefficient_table = pd.DataFrame({
    "Variable": feature_names,
    "Coefficient": coefficients
})

coefficient_table["Coefficient"] = (
    coefficient_table["Coefficient"].round(3)
)

coefficient_table = coefficient_table.sort_values(
    "Coefficient",
    ascending=False
)

print("\n========== REGRESSION COEFFICIENTS ==========")
print(coefficient_table.to_string(index=False))


# ============================================================
# 7. OLS STATISTICAL REGRESSION
# ============================================================

regression_df = pd.get_dummies(
    clean_df[
        numerical_features
        + categorical_features
        + ["Exam_Score"]
    ],
    columns=categorical_features,
    drop_first=True,
    dtype=float
)

X_stats = regression_df.drop(
    columns=["Exam_Score"]
)

y_stats = regression_df["Exam_Score"]

X_stats = sm.add_constant(X_stats)

ols_model = sm.OLS(
    y_stats,
    X_stats
).fit()

print("\n========== OLS REGRESSION RESULTS ==========")
print(ols_model.summary())


# ============================================================
# 8. REGRESSION DIAGNOSTICS
# ============================================================

vif_data = pd.DataFrame()
vif_data["Variable"] = X_stats.columns

vif_data["VIF"] = [
    variance_inflation_factor(
        X_stats.values,
        i
    )
    for i in range(X_stats.shape[1])
]

vif_data["VIF"] = vif_data["VIF"].round(2)

print("\n========== VIF RESULTS ==========")
print(vif_data.to_string(index=False))

residuals = ols_model.resid
fitted_values = ols_model.fittedvalues

print("\n========== RESIDUAL DIAGNOSTICS ==========")
print(
    "Mean residual:",
    round(residuals.mean(), 4)
)
print(
    "Minimum residual:",
    round(residuals.min(), 4)
)
print(
    "Maximum residual:",
    round(residuals.max(), 4)
)


# Residual plot
plt.figure(figsize=(8, 5))

sns.scatterplot(
    x=fitted_values,
    y=residuals,
    alpha=0.4
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Fitted Exam Score")
plt.ylabel("Residual")
plt.title("Residuals vs Fitted Values")

plt.tight_layout()

plt.savefig(
    "outputs/regression_residuals.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. FINAL PROJECT VISUALIZATIONS
# ============================================================

# Consistent plotting style
sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# ----- Visualization 1: Attendance -----

plt.figure(figsize=(8, 5))

sns.regplot(
    data=clean_df,
    x="Attendance",
    y="Exam_Score",
    scatter_kws={"alpha": 0.25},
    line_kws={"linewidth": 2}
)

plt.title("Attendance and Exam Score")
plt.xlabel("Attendance (%)")
plt.ylabel("Exam Score")

plt.tight_layout()

plt.savefig(
    "outputs/visualization_01_attendance_exam_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----- Visualization 2: Study Hours -----

plt.figure(figsize=(8, 5))

sns.regplot(
    data=clean_df,
    x="Hours_Studied",
    y="Exam_Score",
    scatter_kws={"alpha": 0.25},
    line_kws={"linewidth": 2}
)

plt.title("Study Hours and Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")

plt.tight_layout()

plt.savefig(
    "outputs/visualization_02_study_hours_exam_score.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----- Visualization 3: Access to Resources -----

resource_plot = (
    clean_df.groupby("Access_to_Resources")["Exam_Score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=resource_plot.index,
    y=resource_plot.values
)

plt.title(
    "Average Exam Score by Access to Resources"
)
plt.xlabel("Access to Resources")
plt.ylabel("Average Exam Score")

for i, value in enumerate(resource_plot.values):
    ax.text(
        i,
        value + 0.2,
        f"{value:.2f}",
        ha="center"
    )

plt.ylim(0, 75)
plt.tight_layout()

plt.savefig(
    "outputs/visualization_03_access_to_resources.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----- Visualization 4: Parental Involvement -----

involvement_plot = (
    clean_df.groupby("Parental_Involvement")["Exam_Score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=involvement_plot.index,
    y=involvement_plot.values
)

plt.title(
    "Average Exam Score by Parental Involvement"
)
plt.xlabel("Parental Involvement")
plt.ylabel("Average Exam Score")

for i, value in enumerate(involvement_plot.values):
    ax.text(
        i,
        value + 0.2,
        f"{value:.2f}",
        ha="center"
    )

plt.ylim(0, 75)
plt.tight_layout()

plt.savefig(
    "outputs/visualization_04_parental_involvement.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----- Visualization 5: Family Income -----

income_plot = (
    clean_df.groupby("Family_Income")["Exam_Score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=income_plot.index,
    y=income_plot.values
)

plt.title("Average Exam Score by Family Income")
plt.xlabel("Family Income")
plt.ylabel("Average Exam Score")

for i, value in enumerate(income_plot.values):
    ax.text(
        i,
        value + 0.2,
        f"{value:.2f}",
        ha="center"
    )

plt.ylim(0, 75)
plt.tight_layout()

plt.savefig(
    "outputs/visualization_05_family_income.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ----- Visualization 6: Parental Education -----

education_plot = (
    clean_df.groupby(
        "Parental_Education_Level"
    )["Exam_Score"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=education_plot.index,
    y=education_plot.values
)

plt.title(
    "Average Exam Score by Parental Education Level"
)
plt.xlabel("Parental Education Level")
plt.ylabel("Average Exam Score")

for i, value in enumerate(education_plot.values):
    ax.text(
        i,
        value + 0.2,
        f"{value:.2f}",
        ha="center"
    )

plt.ylim(0, 75)
plt.tight_layout()

plt.savefig(
    "outputs/visualization_06_parental_education.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n========== PROJECT ANALYSIS COMPLETE ==========")
print("Dataset:", file_path)
print("Rows:", len(clean_df))
print("Columns:", len(clean_df.columns))
print("R-squared:", round(r2, 3))
print("MAE:", round(mae, 3))
print("Final visualizations saved in the outputs/ folder.")
print("Regression residual plot saved in the outputs/ folder.")
