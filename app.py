import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="German Credit Classification App", layout="wide")

st.title("German Credit Risk Classification")
st.write("View baseline model performance or upload your own test CSV.")

# ======================================================
# 🔹 BASELINE METRICS (Stored Results)
# ======================================================

baseline_metrics = {
    "Logistic Regression": {"Accuracy": 0.745, "Precision": 0.807, "Recall": 0.836, "F1 Score": 0.821, "MCC": 0.379, "AUC": 0.783},
    "Decision Tree": {"Accuracy": 0.720, "Precision": 0.868, "Recall": 0.707, "F1 Score": 0.780, "MCC": 0.423, "AUC": 0.757},
    "KNN": {"Accuracy": 0.710, "Precision": 0.770, "Recall": 0.836, "F1 Score": 0.801, "MCC": 0.271, "AUC": 0.714},
    "Naive Bayes": {"Accuracy": 0.695, "Precision": 0.832, "Recall": 0.707, "F1 Score": 0.764, "MCC": 0.349, "AUC": 0.751},
    "Random Forest": {"Accuracy": 0.775, "Precision": 0.806, "Recall": 0.893, "F1 Score": 0.847, "MCC": 0.431, "AUC": 0.809},
    "XGBoost": {"Accuracy": 0.760, "Precision": 0.799, "Recall": 0.879, "F1 Score": 0.837, "MCC": 0.394, "AUC": 0.789},
}

st.header("Baseline Model Performance")

baseline_model_name = st.selectbox(
    "Select Model (Baseline Results)",
    list(baseline_metrics.keys()),
    key="baseline_dropdown"
)

if st.button("Show Baseline Metrics"):
    metrics = baseline_metrics[baseline_model_name]

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", metrics["Accuracy"])
    col2.metric("Precision", metrics["Precision"])
    col3.metric("Recall", metrics["Recall"])

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", metrics["F1 Score"])
    col5.metric("MCC", metrics["MCC"])
    col6.metric("AUC", metrics["AUC"])


# ======================================================
# 🔹 REAL MODEL EVALUATION (UPLOADED FILE)
# ======================================================

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/Logistic_Regression.pkl"),
        "Decision Tree": joblib.load("model/Decision_Tree.pkl"),
        "KNN": joblib.load("model/KNN.pkl"),
        "Naive Bayes": joblib.load("model/Naive_Bayes.pkl"),
        "Random Forest": joblib.load("model/Random_Forest.pkl"),
        "XGBoost": joblib.load("model/XGBoost.pkl"),
    }

@st.cache_resource
def load_preprocessing():
    scaler = joblib.load("model/scaler.pkl")
    numerical_cols = joblib.load("model/numerical_cols.pkl")
    return scaler, numerical_cols

models = load_models()
scaler, numerical_cols = load_preprocessing()

st.header("Upload Your Own Test File")

uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    target_column = df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    uploaded_model_name = st.selectbox(
        "Select Model (Uploaded Data)",
        list(models.keys()),
        key="uploaded_dropdown"
    )

    model = models[uploaded_model_name]

    if st.button("Run Evaluation on Uploaded Data"):
        with st.spinner("Running model..."):
            time.sleep(2)  # intentional delay

            X_processed = X.copy()

            if uploaded_model_name in ["Logistic Regression", "KNN"]:
                X_processed[numerical_cols] = scaler.transform(X_processed[numerical_cols])

            y_pred = model.predict(X_processed)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_processed)[:, 1]
                auc = roc_auc_score(y, y_prob)
            else:
                auc = None

            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            mcc = matthews_corrcoef(y, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.3f}")
        col5.metric("MCC", f"{mcc:.3f}")
        if auc is not None:
            col6.metric("AUC", f"{auc:.3f}")

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        ax.matshow(cm)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center')

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y, y_pred)
        st.text(report)

else:
    st.info("Upload a CSV file to evaluate a model.")
