from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

RANDOM_STATE = 42
TARGET = "target"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(y, predictions, zero_division=0),
        "Recall": recall_score(y, predictions, zero_division=0),
        "F1": f1_score(y, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y, predictions),
    }


def main():
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # Original sklearn coding is malignant=0 and benign=1.
    # Invert it so the clinically important malignant class is positive (1).
    y = (1 - dataset.target).astype(int).rename(TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": (
            "logistic_regression.joblib",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]),
        ),
        "Decision Tree": (
            "decision_tree.joblib",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
            ]),
        ),
        "KNN": (
            "knn.joblib",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=5)),
            ]),
        ),
        "Naive Bayes": (
            "naive_bayes.joblib",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GaussianNB()),
            ]),
        ),
        "Support Vector Machine": (
            "svm.joblib",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]),
        ),
    }

    rows = []
    for model_name, (filename, model) in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / filename)
        metrics = calculate_metrics(model, X_test, y_test)
        rows.append({"ML Model Name": model_name, **metrics})

    test_data = X_test.copy()
    test_data[TARGET] = y_test
    test_data.to_csv(BASE_DIR / "test_data.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    schema = {
        "target_column": TARGET,
        "positive_class": 1,
        "positive_class_name": "malignant",
        "negative_class": 0,
        "negative_class_name": "benign",
        "feature_columns": list(X.columns),
        "test_rows": len(test_data),
        "random_state": RANDOM_STATE,
    }
    (MODEL_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print("Training complete. Saved five models, test_data.csv, schema.json and metrics.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
