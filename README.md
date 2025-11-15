Predicting Customer Churn in Telecom
Project Overview
This repository contains all analyses and code for a customer churn prediction project using the IBM Telco Customer Churn dataset. Our goal is to build robust machine learning models to predict which telecom customers are likely to churn, with a focus on transparent and reproducible data preparation in line with industry best practices and mentorship feedback.

Project Progress
✅ Data Cleaning & Preprocessing (Completed)
All code, exploration, and cleaning up to the modeling-ready stage have been completed and documented below. Each step is accompanied by in-notebook markdown for clarity and learning.

1. Imports and Setup
We imported essential libraries (pandas, numpy, matplotlib, seaborn, pathlib) to handle data analysis, visualization, and file management.

2. Data Organization
We created dedicated folders:

../data/raw/ for unmodified source data.

../data/processed/ for cleaned, analysis-ready datasets.

This clear separation supports reproducibility and easy handoff.

3. Data Loading
We loaded the IBM Telco Customer Churn dataset from CSV into a pandas DataFrame for processing.

4. Initial Data Inspection
We explored:

Dataset shape and column types (df.info())

Basic statistics (df.describe())

Checked for missing values and potential data type mismatches

5. Whitespace and Formatting Fixes
To avoid subtle errors, we:

Stripped extra spaces from all column names and string cells.

6. Handling Missing Values
We addressed missing/invalid data by:

Converting TotalCharges to a numeric type.

Filling any missing TotalCharges with the median value.

7. Removing Unnecessary Columns
We dropped unique identifier columns (customerID) that do not add predictive value for modeling.

8. Feature Engineering
We created new, more insightful features for modeling, such as:

Binning tenure into tenure groups to better capture customer lifetime patterns.

9. Encoding Categorical Variables
All categorical fields were transformed using one-hot encoding, resulting in a fully numeric dataset suitable for machine learning algorithms.

10. Quality Checks
We performed rigorous final checks before saving:

Verified no missing or infinite values remain

Confirmed no duplicate records exist

Explored target variable balance (churned vs. not churned)

11. Saving the Cleaned Dataset
The cleaned, encoded data was saved at:
../data/processed/telco_cleaned.csv
This file will be used for model training and further analysis.

Next Steps
Exploratory Data Analysis (EDA): deeper visual insights, correlation checks, etc.

Model training and evaluation.

Reproducibility
All code cells are clearly numbered and documented throughout the notebook.
For library version control, run:

python
import sys
print("Python:", sys.version)
print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)
