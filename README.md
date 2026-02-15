# Multiple Classification Models on the German Credit Risk Dataset

---

## a. Problem Statement

This project aims to build different machine learning classification models and to deploy them by creating an interactive Streamlit app. The app shows the performance of the models and allows the user to upload a test dataset to check the metrics and compare the models. The metrics used are :
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coefficient (MCC Score).

The models are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier - Gaussian or Multinomial
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
| Logistic Regression | The Logistic Regression model has a balanced performance with a good Recall (0.836) and a strong F1 (0.821). However, it has a moderate MCC (0.379) and did not top in any metric. Overall, it is a stable and reliable baseline model which detects most risky borrowers while keeping a balance. |
| Decision Tree | Achieved the highest precision (0.868), but has a lower recall (0.707). In a credit risk classification, missing actual defaulters (low recall) is risky as it can lead to financial loss. Therefore, this model is not ideal for this problem. |
| kNN | Showed good recall, (0.836), but lower overall accuracy and MCC. It detects many risky borrowers but the overall predictive strength is weak and less stable. |
| Naive Bayes | Lowest accuracy among models (0.695). It is a simple probabilistic model which is acceptable but weaker than others in predictive power. |
| Random Forest (Ensemble) | Random Forest was the best overall performer with the highest Accuracy (0.775), the highest AUC (0.809), the highest Recall (0.893), the highest F1 (0.847) and the highest MCC (0.431). It detects the riskiest borrowers while maintaining strong an overall balance and robustness. It also minimizes costly false negatives. |
| XGBoost (Ensemble) | Strong performance close to Random Forest. Slightly lower accuracy but robust and powerful gradient boosting method. |

---

Among all the implemented models, **Random Forest** achieved the best overall performance across most evaluation metrics, including Accuracy, F1 Score, and MCC. The ensemble methods outperformed the individual models, indicating that combining multiple weak learners improves predictive capability for credit risk classification.
