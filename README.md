# Multiple Classification Models on the German Credit Risk Dataset

---

## a. Problem Statement

This project aims to build different machine learning classification models and to deploy them by creating an interactive Streamlit app. The app shows the performance of the models and allows the user to upload a test dataset to check the metrics and compare the models. The metrics used are :
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score).

The models are:
1. Logistic Regression
2. Decision Tree Classifi er
3. K-Nearest Neighbor Classifi er
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

All models are trained on the same dataset, the UCI Statlog (German Credit Data) dataset, and they predict whether a loan applicant is a good or bad credit risk based on financial and demographic attributes. This is a binary classification problem as it classifies the loan applicants into the two classes - good and bad. The goal is to evaluate model performance using multiple evaluation metrics.

---

## b. Dataset Description

The dataset used is the UCI Statlog (German Credit Data) dataset.

- Total Instances: 1000
- Total Features: 20 input features
- Target Variable:
  - 1 → Good Credit Risk
  - 2 → Bad Credit Risk
  The target values are converted to binary format ( 1 and 0)

The dataset contains a mix of categorical and numerical features including:
- Credit history
- Loan duration
- Credit amount
- Employment status
- Age
- Housing
- Savings account status
- Personal status

Some preprocessing has been performed on the dataset:
- Label encoding of categorical variables
- Feature scaling for models requiring normalization (Logistic Regression, kNN)
- Train-test split (80% training, 20% testing)

---

## c. Models Used and Performance Comparison

Six machine learning models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors (kNN)  
4. Naive Bayes  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|------|----------|--------|----------|------|
| Logistic Regression | 0.745 | 0.783 | 0.807 | 0.836 | 0.821 | 0.379 |
| Decision Tree | 0.720 | 0.757 | 0.868 | 0.707 | 0.780 | 0.423 |
| kNN | 0.710 | 0.714 | 0.770 | 0.836 | 0.801 | 0.271 |
| Naive Bayes | 0.695 | 0.751 | 0.832 | 0.707 | 0.764 | 0.349 |
| Random Forest (Ensemble) | 0.775 | 0.809 | 0.806 | 0.893 | 0.847 | 0.431 |
| XGBoost (Ensemble) | 0.760 | 0.789 | 0.799 | 0.879 | 0.837 | 0.394 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|--------------------------------------|
| Logistic Regression | Performed well with balanced precision and recall. Provides stable and interpretable baseline results. Suitable for linear relationships in credit risk prediction. |
| Decision Tree | Achieved high precision but lower recall. It tends to overfit and may not generalize as well as ensemble methods. |
| kNN | Showed good recall but lower overall accuracy and MCC. Performance depends heavily on feature scaling and distance metric. |
| Naive Bayes | Lowest accuracy among models. Assumption of feature independence may not hold for this dataset, limiting performance. |
| Random Forest (Ensemble) | Best overall performer. Highest accuracy, F1 score, and MCC. Handles non-linearity well and reduces overfitting compared to Decision Tree. |
| XGBoost (Ensemble) | Strong performance close to Random Forest. Slightly lower accuracy but robust and powerful gradient boosting method. |

---

Among all the implemented models, **Random Forest** achieved the best overall performance across most evaluation metrics, including Accuracy, F1 Score, and MCC. Ensemble methods outperformed individual models, indicating that combining multiple weak learners improves predictive capability for credit risk classification.
