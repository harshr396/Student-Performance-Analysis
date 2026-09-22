# Student Performance Analysis

## Project Overview

This project investigates the social, economic and educational factors associated with differences in student academic performance.

### Research Question

> What social, economic and educational factors are associated with differences in student academic performance?

The project applies data cleaning, exploratory data analysis (EDA), visualization and multiple linear regression to the **Student Performance Factors Dataset**.

## Dataset

**Dataset:** Student Performance Factors  
**Author:** Ayesha Saher  
**Source:** Kaggle  
**Dataset link:** https://www.kaggle.com/datasets/ayeshaseherr/student-performance

The dataset used in this project contains:

- **6,607 observations**
- **20 variables**

The main outcome variable is `Exam_Score`.

## Objectives

- Examine patterns in student academic performance.
- Analyse the association between attendance and Exam Score.
- Analyse the association between study hours and Exam Score.
- Examine differences in performance by access to educational resources.
- Examine differences in performance by parental involvement.
- Examine the relationship between family income and Exam Score.
- Examine the relationship between parental education and Exam Score.
- Use multiple linear regression to examine several factors simultaneously.
- Translate the analysis into observations, insights, hypotheses and recommendations.

## Technologies Used

- Python
- Spyder
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Statsmodels

## Project Structure

```text
Student-Performance-Analysis/
│
├── README.md
├── requirements.txt
├── HarshRaj_StudentPerformanceAnalysis.py
├── HarshRaj_StudentPerformanceAnalysis_Report.docx
│
├── StudentPerformanceFactors.csv
│ 
│
└── outputs/
    ├── visualization_01_attendance_exam_score.png
    ├── visualization_02_study_hours_exam_score.png
    ├── visualization_03_access_to_resources.png
    ├── visualization_04_parental_involvement.png
    ├── visualization_05_family_income.png
    └── visualization_06_parental_education.png
```

## Analysis Workflow

The project follows a data analytics workflow:

1. Load the raw dataset.
2. Inspect dimensions, variables and data types.
3. Check missing values and duplicate records.
4. Clean categorical missing values using mode imputation.
5. Conduct exploratory data analysis.
6. Create visualizations to investigate relationships with Exam Score.
7. Fit a multiple linear regression model.
8. Examine statistical significance using OLS regression.
9. Check multicollinearity using VIF.
10. Perform an outlier sensitivity check.
11. Interpret observations and insights.
12. Develop hypotheses and recommendations.

## Data Cleaning

The raw dataset contained missing values in:

- `Teacher_Quality`: 78
- `Parental_Education_Level`: 90
- `Distance_from_Home`: 67

These variables are categorical, so missing values were filled using the mode of each variable.

There were no duplicate records.

One `Exam_Score` value was 101, which appears unusual for a conventional 0–100 score. A sensitivity check found that removing this observation changed the mean Exam Score from **67.236 to 67.231**, so the observation was retained and documented as a potential data-quality anomaly.

## Exploratory Data Analysis

Six final visualizations examine:

1. Attendance and Exam Score
2. Study Hours and Exam Score
3. Access to Educational Resources and Exam Score
4. Parental Involvement and Exam Score
5. Family Income and Exam Score
6. Parental Education Level and Exam Score

### Key Results

- Attendance and Exam Score correlation: **0.581**
- Study Hours and Exam Score correlation: **0.445**
- Average Exam Score by Access to Resources:
  - High: **68.09**
  - Medium: **67.13**
  - Low: **66.20**
- Average Exam Score by Parental Involvement:
  - High: **68.09**
  - Medium: **67.10**
  - Low: **66.36**
- Average Exam Score by Family Income:
  - High: **67.84**
  - Medium: **67.33**
  - Low: **66.85**
- Average Exam Score by Parental Education:
  - Postgraduate: **67.97**
  - College: **67.32**
  - High School: **66.90**

These results describe associations in the dataset and should not be interpreted as proof of causation.

## Multiple Linear Regression

The regression model includes:

### Numerical predictors

- Attendance
- Hours Studied
- Previous Scores
- Tutoring Sessions
- Physical Activity

### Categorical predictors

- Family Income
- Parental Education Level
- Access to Resources
- Parental Involvement
- Motivation Level

### Model Results

- **R-squared:** 0.691
- **Adjusted R-squared:** 0.690
- **Mean Absolute Error:** 0.831
- All included predictors had **p < 0.001** in the OLS specification.
- VIF values ranged from **1.00 to 1.86**, indicating no substantial multicollinearity among the included predictors.

The regression was fitted and evaluated on the same observations, so its R-squared and MAE should not be interpreted as out-of-sample predictive performance.

## Important Limitations

- The dataset is observational, so the analysis cannot establish causal relationships.
- One Exam Score value of 101 was identified as a potential data-quality anomaly.
- Residual diagnostics showed a subset of observations with large positive residuals, indicating that the linear model does not fully capture the outcome distribution.
- The regression model was evaluated on the same observations used for fitting.
- Results should not automatically be generalized to all students or educational systems.

## How to Run the Project

1. Install Python.
2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Place StudentPerformanceFactors.csv in the same folder as the Python script.
4. Open `HarshRaj_StudentPerformanceAnalysis.py` in Spyder or another Python IDE.
5. Run the script.
6. The six final visualizations will be saved in the `outputs/` folder.

## Project Deliverables

- Python analysis script
- `requirements.txt`
- Project report in DOCX format
- README documentation
- Visualizations
- Dataset source documentation

## Author

**Harsh Raj**

AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026
