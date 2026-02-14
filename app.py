import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(page_title="Credit Risk Classification")

st.title("German Credit Risk Classification App")
st.write(
    "Select a model to view baseline performance and optionally evaluate "
    "on an uploaded test dataset."
)

# -------------------------------------------------
# Model Selection
# -------------------------------------------------
model_options = {
    "Logistic Regression": "logistic.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

selected_model = st.selectbox("Select Model", list(model_options.keys()))

# -------------------------------------------------
# Baseline Metrics (Original Held-Out Test Set)
# -------------------------------------------------
baseline_metrics = {
    "Logistic Regression": {"Accuracy": 0.745, "Precision": 0.807, "Recall": 0.836, "F1 Score": 0.821, "MCC": 0.379, "AUC": 0.783},
    "Decision Tree": {"Accuracy": 0.720, "Precision": 0.868, "Recall": 0.707, "F1 Score": 0.780, "MCC": 0.423, "AUC": 0.757},
    "KNN": {"Accuracy": 0.710, "Precision": 0.770, "Recall": 0.836, "F1 Score": 0.801, "MCC": 0.271, "AUC": 0.714},
    "Naive Bayes": {"Accuracy": 0.695, "Precision": 0.832, "Recall": 0.707, "F1 Score": 0.764, "MCC": 0.349, "AUC": 0.751},
    "Random Forest": {"Accuracy": 0.775, "Precision": 0.806, "Recall": 0.893, "F1 Score": 0.847, "MCC": 0.431, "AUC": 0.809},
    "XGBoost": {"Accuracy": 0.760, "Precision": 0.799, "Recall": 0.879, "F1 Score": 0.837, "MCC": 0.394, "AUC": 0.789},
}

st.subheader("Baseline Performance (Original Test Set)")
metrics = baseline_metrics[selected_model]

for key, value in metrics.items():
    st.write(f"**{key}:** {value}")

# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.subheader("Evaluate on Uploaded Test Dataset")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    with st.spinner("Evaluating model on uploaded dataset..."):
        time.sleep(2)  # intentional delay for better UX

        df = pd.read_csv(uploaded_file)

        # Separate features and target
        X = df.drop("Target", axis=1)
        y = df["Target"]

        # Load model
        model_path = model_options[selected_model]
        model = joblib.load(model_path)

        # Load scaler and numerical columns
        scaler = joblib.load("scaler.pkl")
        numerical_cols = joblib.load("numerical_cols.pkl")

        # Apply scaling only for required models
        if selected_model in ["Logistic Regression", "KNN"]:
            X[numerical_cols] = scaler.transform(X[numerical_cols])

        # Predictions
        y_pred = model.predict(X)

        # AUC calculation
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, y_proba)
        else:
            auc = "N/A"

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        mcc = matthews_corrcoef(y, y_pred)

    st.success("Evaluation Complete!")

    # -------------------------------------------------
    # Display Metrics
    # -------------------------------------------------
    st.subheader("Performance on Uploaded Dataset")

    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")
    st.write(f"**MCC:** {mcc:.3f}")
    st.write(f"**AUC:** {auc if auc == 'N/A' else round(auc, 3)}")

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    plt.colorbar(cax)

    for (i, j), val in enumerate(cm.flatten()):
        ax.text(j, i, val, ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report
    # -------------------------------------------------
    st.subheader("Classification Report")

    report = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

else:
    st.info("Upload a test dataset to evaluate the selected model.")
