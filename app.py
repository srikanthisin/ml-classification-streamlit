from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Classification Model Evaluator", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_schema():
    return json.loads((MODEL_DIR / "schema.json").read_text(encoding="utf-8"))


def evaluate(model, X, y):
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y, prediction),
        "AUC": roc_auc_score(y, probability),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
        "MCC": matthews_corrcoef(y, prediction),
    }
    return metrics, prediction


schema = load_schema()
required_features = schema["feature_columns"]

st.title("Multiple Classification Model Evaluator")
st.caption("Upload labelled test data, choose a saved classifier, and evaluate it without retraining.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload test CSV", type=["csv"])
    selected_model = st.selectbox("Select model", list(MODEL_FILES))
    st.info("Target coding: 1 = malignant, 0 = benign")
    with open(BASE_DIR / "test_data.csv", "rb") as sample:
        st.download_button(
            "Download sample test data",
            data=sample,
            file_name="test_data.csv",
            mime="text/csv",
        )

if uploaded_file is None:
    st.subheader("Expected CSV format")
    st.write(f"The CSV must contain {len(required_features)} feature columns and the `{TARGET}` column.")
    st.dataframe(pd.read_csv(BASE_DIR / "test_data.csv").head(), use_container_width=True)
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [column for column in required_features + [TARGET] if column not in data.columns]
extra = [column for column in data.columns if column not in required_features + [TARGET]]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()
if data[TARGET].isna().any():
    st.error("The target column contains missing values.")
    st.stop()
if not set(data[TARGET].unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()
if data[TARGET].nunique() < 2:
    st.error("AUC requires both target classes. Upload test data containing 0 and 1.")
    st.stop()

X_test = data[required_features]
y_test = data[TARGET].astype(int)
model = load_model(MODEL_DIR / MODEL_FILES[selected_model])

try:
    metrics, predictions = evaluate(model, X_test, y_test)
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")
    st.stop()

st.success(f"Evaluated {selected_model} on {len(data)} test records. No retraining was performed.")
if extra:
    st.warning("Ignored extra columns: " + ", ".join(extra))

st.subheader("Evaluation Metrics")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Benign (0)", "Malignant (1)"],
                yticklabels=["Benign (0)", "Malignant (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Uploaded data preview"):
    st.dataframe(data.head(20), use_container_width=True)
