# loan-approval-ml
LOAN APPROVAL PREDICTION USING SUPERVISED MACHINE LEARNING
==========================================================

Author: Jorge López Díaz
Date: December 2025

----------------------------------------------------------
1. PROJECT OVERVIEW
----------------------------------------------------------

This project addresses a binary classification problem in the financial sector: 
predicting whether a loan application should be approved or rejected.

Using a realistic dataset with personal, employment, and financial information 
from applicants in the US and Canada, several supervised machine learning models 
are developed, evaluated, and compared to identify the most suitable approach 
for minimizing financial risk.

The main objective is not only high accuracy, but reducing costly financial 
errors, especially false positives (approving loans that later default).

----------------------------------------------------------
2. DATASET
----------------------------------------------------------

Dataset source:
Realistic Loan Approval Dataset (US & Canada)
https://www.kaggle.com/datasets/parthpatel2130/realistic-loan-approval-dataset-us-and-canada

Target variable:
- loan_status
    1 → Loan Approved
    0 → Loan Rejected

Original dataset size:
- 50,000 observations

Sample used in this project:
- 5,000 observations (stratified sampling)

----------------------------------------------------------
3. METHODOLOGY
----------------------------------------------------------

The project follows a structured machine learning workflow:

1. Data cleaning and preprocessing
2. Feature selection and transformation
3. Model training and hyperparameter tuning
4. Model evaluation and comparison
5. Final model selection based on financial risk criteria

----------------------------------------------------------
4. DATA PREPROCESSING
----------------------------------------------------------

Key preprocessing steps:

- Removal of non-predictive identifiers
- Stratified train-test split (80% train / 20% test)
- Feature selection to avoid multicollinearity
- ColumnTransformer pipeline including:
    * Log transformation + standardization for skewed variables
    * Standard scaling for numerical features
    * One-hot encoding for categorical variables
- Prevention of data leakage using pipelines

----------------------------------------------------------
5. MODELS IMPLEMENTED
----------------------------------------------------------

The following supervised learning models were trained and optimized using 
cross-validation and GridSearch:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (Linear)
- Support Vector Machine (Non-linear, RBF kernel)
- Decision Tree
- Logistic Regression

Class imbalance was handled using:
- SMOTE (for KNN)
- class_weight='balanced' (for SVM, Decision Tree, Logistic Regression)

----------------------------------------------------------
6. EVALUATION METRIC
----------------------------------------------------------

Primary optimization metric:
- F0.7 Score

Reason:
- Precision is prioritized over recall to minimize false positives
- Approving a bad loan is more costly than rejecting a good one

This metric aligns the model performance with real financial risk management.

----------------------------------------------------------
7. RESULTS AND MODEL SELECTION
----------------------------------------------------------

Cross-validation F0.7 scores:

- Non-linear SVM (RBF): 0.8945
- Logistic Regression: 0.8607
- Linear SVM: 0.8595
- Decision Tree: 0.8460
- KNN: 0.8475

Best model:
- Support Vector Machine with RBF kernel

Reasons:
- Highest F0.7 score
- Strong generalization performance
- Effective reduction of false positives
- Balanced trade-off between risk protection and opportunity cost

----------------------------------------------------------
8. INTERPRETABILITY
----------------------------------------------------------

Model interpretability was analyzed using:

- SHAP values (Linear SVM and Logistic Regression)
- Permutation Feature Importance (Non-linear SVM)
- Decision Tree visualization and rule interpretation

Key influential features include:
- Credit score
- Debt-to-income ratio
- Payment-to-income ratio
- Credit history length

----------------------------------------------------------
9. TECHNOLOGIES USED
----------------------------------------------------------

- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- SHAP
- Matplotlib / Seaborn
- Jupyter Notebook

----------------------------------------------------------
10. CONCLUSION
----------------------------------------------------------

This project demonstrates how machine learning models can be aligned with 
real-world financial objectives, prioritizing risk minimization over raw accuracy.

The selected non-linear SVM provides a robust and practical solution for loan 
approval decision support systems.

----------------------------------------------------------

