# Loan Approval Prediction using Supervised Machine Learning

**Author:** Jorge López Díaz · **Date:** December 2025

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

## Table of contents

1. [Project overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Methodology](#3-methodology)
4. [Data preprocessing](#4-data-preprocessing)
5. [Models implemented](#5-models-implemented)
6. [Evaluation metric](#6-evaluation-metric)
7. [Results and model selection](#7-results-and-model-selection)
8. [Interpretability](#8-interpretability)
9. [Technologies used](#9-technologies-used)
10. [Conclusion](#10-conclusion)
11. [Repository structure](#repository-structure)
12. [How to run](#how-to-run)

---

## 1. Project overview

This project addresses a binary classification problem in the financial sector: predicting whether a loan application should be approved or rejected.

Using a realistic dataset with personal, employment, and financial information from applicants in the US and Canada, several supervised machine learning models are developed, evaluated, and compared to identify the most suitable approach for minimizing financial risk.

The main objective is not only high accuracy, but reducing costly financial errors, especially false positives (approving loans that later default).

## 2. Dataset

**Source:** [Realistic Loan Approval Dataset (US & Canada)](https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada)

**Target variable:** `loan_status`

- `1` → Loan Approved
- `0` → Loan Rejected

| | Observations |
|---|---|
| Original dataset size | 50,000 |
| Sample used in this project | 5,000 (stratified sampling) |

## 3. Methodology

The project follows a structured machine learning workflow:

1. Data cleaning and preprocessing
2. Feature selection and transformation
3. Model training and hyperparameter tuning
4. Model evaluation and comparison
5. Final model selection based on financial risk criteria

## 4. Data preprocessing

Key preprocessing steps:

- Removal of non-predictive identifiers
- Stratified train-test split (80% train / 20% test)
- Feature selection to avoid multicollinearity
- `ColumnTransformer` pipeline including:
  - Log transformation + standardization for skewed variables
  - Standard scaling for numerical features
  - One-hot encoding for categorical variables
- Prevention of data leakage using pipelines

## 5. Models implemented

The following supervised learning models were trained and optimized using cross-validation and GridSearch:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear)
- Support Vector Machine (Non-linear, RBF kernel)
- Decision Tree
- Logistic Regression

Class imbalance was handled using:

- SMOTE (for KNN)
- `class_weight='balanced'` (for SVM, Decision Tree, Logistic Regression)

## 6. Evaluation metric

**Primary optimization metric:** F0.7 Score

Reason:

- Precision is prioritized over recall to minimize false positives
- Approving a bad loan is more costly than rejecting a good one

This metric aligns the model performance with real financial risk management.

## 7. Results and model selection

Cross-validation F0.7 scores:

| Model | F0.7 (CV) |
|---|---|
| **Non-linear SVM (RBF)** | **0.8945** |
| Logistic Regression | 0.8607 |
| Linear SVM | 0.8595 |
| KNN | 0.8475 |
| Decision Tree | 0.8460 |

**Best model:** Support Vector Machine with RBF kernel

Reasons:

- Highest F0.7 score
- Strong generalization performance
- Effective reduction of false positives
- Balanced trade-off between risk protection and opportunity cost

## 8. Interpretability

Model interpretability was analyzed using:

- SHAP values (Linear SVM and Logistic Regression)
- Permutation Feature Importance (Non-linear SVM)
- Decision Tree visualization and rule interpretation

Key influential features include:

- Credit score
- Debt-to-income ratio
- Payment-to-income ratio
- Credit history length

## 9. Technologies used

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- SHAP
- Matplotlib / Seaborn
- Jupyter Notebook

## 10. Conclusion

This project demonstrates how machine learning models can be aligned with real-world financial objectives, prioritizing risk minimization over raw accuracy.

The selected non-linear SVM provides a robust and practical solution for loan approval decision support systems.

---

## Repository structure

```text
loan-approval-ml/
├── data/
│   └── raw/
│       └── Loan_approval_data_2025.csv      # Sampled dataset (5,000 observations)
├── notebooks/
│   └── loan_approval.ipynb                  # Full analysis: preprocessing, models, evaluation, SHAP
├── reports/
│   └── loan_approval_general_overview.pdf   # General overview report
├── requirements.txt                         # Python dependencies
├── .gitignore
└── README.md
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/jlopezd-DataScienceandAI/loan-approval-ml.git
cd loan-approval-ml

# 2. Create and activate a virtual environment (Python 3.11 recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/loan_approval.ipynb
```

> The notebook reads the dataset from `../data/raw/`, so it should be opened from the `notebooks/` folder (Jupyter does this automatically).
