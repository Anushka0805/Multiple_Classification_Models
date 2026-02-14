import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
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
st.write("Evaluate baseline data or upload your own test CSV.")

# -----------------------------
# Load Models and Preprocessing Objects
# -----------------------------
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


# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, X, y, model_name):
    X_processed = X.copy()

    if model_name in ["Logistic Regression", "KNN"]:
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


# ======================================================
# 🔹 SECTION 1 — BASELINE DATA
# ======================================================
st.header("Baseline Evaluation")

baseline_model_name = st.selectbox(
    "Select Model for Baseline",
    list(models.keys()),
    key="baseline_model"
)

baseline_model = models[baseline_model_name]

baseline_df = pd.read_csv("baseline_test.csv")  # <-- your baseline CSV
target_column = baseline_df.columns[-1]

X_base = baseline_df.drop(columns=[target_column])
y_base = baseline_df[target_column]

if st.button("Run Baseline Evaluation"):
    with st.spinner("Evaluating baseline data..."):
        evaluate_model(baseline_model, X_base, y_base, baseline_model_name)


# ======================================================
# 🔹 SECTION 2 — UPLOADED TEST FILE
# ======================================================
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
        "Select Model for Uploaded Data",
        list(models.keys()),
        key="uploaded_model"
    )

    uploaded_model = models[uploaded_model_name]

    if st.button("Run Uploaded Data Evaluation"):
        with st.spinner("Evaluating uploaded data..."):
            evaluate_model(uploaded_model, X, y, uploaded_model_name)
