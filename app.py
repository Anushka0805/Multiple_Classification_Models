import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="German Credit Classification", layout="wide")

st.title("German Credit Risk Classification App")

st.markdown("Upload a test dataset (CSV format) to evaluate selected model.")

# -------------------------
# Load Saved Objects
# -------------------------
@st.cache_resource
def load_artifacts(model_name):
    model = joblib.load(f"{model_name}.pkl")
    scaler = joblib.load("scaler.pkl")
    numerical_cols = joblib.load("numerical_cols.pkl")
    return model, scaler, numerical_cols


# -------------------------
# Model Selection
# -------------------------
model_options = [
    "Logistic_Regression",
    "Decision_Tree",
    "KNN",
    "Naive_Bayes",
    "Random_Forest",
    "XGBoost"
]

selected_model = st.selectbox("Select Model", model_options)


# -------------------------
# File Upload
# -------------------------
uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    df_test = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.write(df_test.head())

    if "Target" not in df_test.columns:
        st.error("Uploaded CSV must contain 'Target' column.")
    else:

        X_test = df_test.drop("Target", axis=1)
        y_test = df_test["Target"]

        # Load model artifacts
        model, scaler, numerical_cols = load_artifacts(selected_model)

        # Apply scaling only if needed
        if selected_model in ["Logistic_Regression", "KNN"]:
            X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        st.table(metrics_df)

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
