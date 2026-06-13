# Customer Churn Prediction — Tabular Machine Learning Project

## Overview

This project predicts customer churn using a tabular machine learning pipeline built with Python, pandas, and scikit-learn.

The goal is to build a clean and reproducible junior-level machine learning project that covers the main steps of a real classification workflow: data loading, data cleaning, exploratory data analysis, preprocessing, model training, evaluation, feature importance, and model persistence.

## Problem Statement

Customer churn occurs when a customer stops using a company's service. For subscription-based businesses, identifying customers who are likely to churn can help teams design better retention strategies.

This project frames churn prediction as a binary classification problem:

* `0`: customer did not churn
* `1`: customer churned

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer-level information such as:

* Demographic information
* Account information
* Subscribed services
* Monthly and total charges
* Churn label

The target variable is `Churn`.

## Tech Stack

* Python
* pandas
* scikit-learn
* matplotlib
* joblib

## Methodology

The project follows these steps:

1. Download and load the dataset.
2. Clean the data:

   * Convert `TotalCharges` to numeric.
   * Convert the churn target from `Yes` / `No` to `1` / `0`.
   * Remove duplicate rows.
3. Perform basic exploratory data analysis:

   * Target distribution.
   * Churn rate by contract type.
   * Monthly charges by churn status.
4. Split the data into training and test sets using stratified sampling.
5. Preprocess the data:

   * Median imputation for numeric features.
   * Most frequent imputation for categorical features.
   * Standard scaling for numeric features.
   * One-hot encoding for categorical features.
6. Train two models:

   * Logistic Regression
   * Random Forest Classifier
7. Evaluate models using:

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * ROC-AUC
8. Select the best model based on ROC-AUC.
9. Save:

   * Best model with joblib.
   * Metrics as JSON.
   * Confusion matrix.
   * Feature importance.

## Results

The final models were evaluated on the test set using accuracy, precision, recall, F1-score, and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |
| Random Forest | 0.7601 | 0.5361 | 0.7139 | 0.6124 | 0.8351 |

Best model selected by ROC-AUC:

```text
Logistic Regression
```

Logistic Regression achieved the highest ROC-AUC score. Although Random Forest had slightly higher accuracy and precision, Logistic Regression performed better at ranking churn risk overall and had higher recall, which is useful in churn detection because missing likely churners can be costly.

Generated artifacts:

* `reports/metrics.json`
* `reports/feature_importance.csv`
* `reports/figures/confusion_matrix.png`
* `reports/figures/feature_importance.png`
* `models/best_model.joblib`

## How to Run

### 1. Clone the repository

```bash
git clone <https://github.com/Rocnarx/customer-churn-prediction>
cd customer-churn-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On macOS / Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run EDA

```bash
python -m src.eda
```

### 5. Train and evaluate models

```bash
python -m src.train
```

## Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Possible improvements:

* Add cross-validation.
* Add hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`.
* Compare additional models such as Gradient Boosting or XGBoost.
* Tune the classification threshold based on business goals.
* Add SHAP or permutation importance for deeper explainability.
* Build a small prediction API with FastAPI.
* Add unit tests for data cleaning and preprocessing.
