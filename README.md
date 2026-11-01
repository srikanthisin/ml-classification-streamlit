# Multiple Classification Models with Streamlit

An end-to-end machine learning classification project that trains and evaluates five classification algorithms on the same dataset, saves the trained models, and deploys an interactive Streamlit application for evaluating the saved models on held-out test data without retraining.

## Problem Statement

The objective of this project is to implement, evaluate, compare, and deploy five machine learning classification models using a common public classification dataset.

The following classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

Each model is evaluated using the following metrics:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The trained models are saved as `.joblib` files and used by an interactive Streamlit application. The Streamlit application accepts a labelled test CSV, loads the selected saved model, generates predictions, and displays evaluation metrics, a confusion matrix, and a classification report.

The Streamlit application does not retrain the models on the uploaded test data.

---

## Dataset Description

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository, distributed through `sklearn.datasets.load_breast_cancer`
- **Problem Type:** Binary Classification
- **Total Instances:** 569
- **Predictive Features:** 30 numerical features
- **Minimum assignment requirement:** 500 instances and 12 features
- **Training/Test Split:** 75% training and 25% testing
- **Training Instances:** 426
- **Test Instances:** 143
- **Split Strategy:** Stratified train-test split with `random_state=42`
- **Target Encoding Used:** `1 = Malignant`, `0 = Benign`

The dataset therefore satisfies the assignment requirement of a minimum of 500 instances and 12 predictive features.

The complete dataset contains **569 instances**. A stratified 75:25 train-test split is applied, resulting in **426 training instances** and **143 testing instances**.

The five machine learning models are trained only on the 426-instance training partition.

The file `test_data.csv` contains only the **143 held-out test instances** together with the target column. This file is uploaded to the Streamlit application for evaluation of the previously trained models.

The models are **not retrained** using `test_data.csv`.

The original scikit-learn representation encodes malignant as `0` and benign as `1`. For this project, the coding is reversed so that:

- `1 = Malignant`
- `0 = Benign`

Therefore, Precision, Recall, and F1 Score directly describe the model's ability to detect malignant cases.

---

## GitHub Repository Link

[GitHub Repository](https://github.com/srikanthisin/ml-classification-streamlit)

---

## Live Streamlit App Link

**To be updated after deployment on Streamlit Community Cloud.**

`https://ml-classification-app-srsbx2a9wq8duyannle9cv.streamlit.app/`

---

## Models Used

The following five classification models are trained on the same training dataset:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (KNN) Classifier**
4. **Gaussian Naive Bayes Classifier**
5. **Support Vector Machine (SVM)**

Preprocessing and the corresponding estimator are stored together using scikit-learn pipelines.

Median-value imputation is included in each pipeline to handle missing values if encountered during prediction.

Feature standardization using `StandardScaler` is used for:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

Scaling is particularly important for distance-based and margin-based algorithms such as KNN and SVM.

All five trained pipelines are saved as `.joblib` files and later loaded by the Streamlit application.

---

## Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9650 | 0.9962 | 0.9800 | 0.9245 | 0.9515 | 0.9251 |
| Decision Tree | 0.9580 | 0.9473 | 0.9796 | 0.9057 | 0.9412 | 0.9103 |
| KNN | 0.9580 | 0.9860 | 0.9796 | 0.9057 | 0.9412 | 0.9103 |
| Naive Bayes | 0.9441 | 0.9952 | 1.0000 | 0.8491 | 0.9184 | 0.8830 |
| **Support Vector Machine** | **0.9720** | **0.9966** | **1.0000** | **0.9245** | **0.9608** | **0.9408** |

---

## Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression achieved strong overall performance with an Accuracy of 0.9650, AUC of 0.9962, F1 Score of 0.9515, and MCC of 0.9251. The malignant-class recall of 0.9245 indicates that most malignant cases were correctly identified. It was the second-best overall performer and produced results close to SVM. |
| **Decision Tree** | Decision Tree achieved an Accuracy of 0.9580, F1 Score of 0.9412, and MCC of 0.9103. However, its AUC of 0.9473 was the lowest among the five models, indicating weaker probability-ranking performance compared with the other classifiers. |
| **KNN** | KNN achieved an Accuracy of 0.9580, AUC of 0.9860, F1 Score of 0.9412, and MCC of 0.9103. Its classification performance on the selected test set was similar to the Decision Tree, while its AUC was noticeably stronger. Feature standardization was applied before KNN because the algorithm is distance-based. |
| **Naive Bayes** | Gaussian Naive Bayes achieved perfect Precision of 1.0000 and a high AUC of 0.9952. However, its Recall was 0.8491, the lowest among the five classifiers. This means it generated no false-positive malignant predictions on this test set but missed more malignant cases than the other models. |
| **Support Vector Machine** | Support Vector Machine produced the strongest overall performance, achieving Accuracy of 0.9720, AUC of 0.9966, Precision of 1.0000, F1 Score of 0.9608, and MCC of 0.9408. It achieved the highest Accuracy, AUC, F1, and MCC among the evaluated models while maintaining malignant-class Recall of 0.9245. |
| **Overall Winner** | **Support Vector Machine (SVM)** is selected as the overall best-performing classifier for this dataset. It achieved the highest Accuracy (0.9720), AUC (0.9966), F1 Score (0.9608), and MCC (0.9408), together with perfect Precision (1.0000). |

### Overall Model Performance

Based on the test-set evaluation, **Support Vector Machine is the overall winner**.

Although Accuracy is useful, the final comparison is not based only on Accuracy. MCC, AUC, F1 Score, Recall, and the confusion matrix are also considered because they provide a more complete indication of binary classification performance.

Logistic Regression was a close second and also demonstrated excellent classification performance.

Naive Bayes achieved perfect Precision but comparatively lower Recall, showing the trade-off between avoiding false-positive malignant predictions and detecting all malignant cases.

---

## Repository Structure

```text
ml-classification-streamlit/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── schema.json
└── results/
    └── model_metrics.csv
```

### Important Files

- `train_models.py` loads the dataset, performs the train-test split, trains all five models, calculates evaluation metrics, and saves the trained pipelines.
- `app.py` implements the interactive Streamlit application.
- `test_data.csv` contains the 143 held-out test instances and target values.
- `model/` contains the five saved trained model pipelines.
- `model/schema.json` contains feature and target metadata required by the application.
- `results/model_metrics.csv` contains the evaluation results for all five classifiers.
- `requirements.txt` contains the Python packages required to run and deploy the application.

---

## Model Training and Evaluation Workflow

The project follows the workflow below:

1. Load the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Convert the target coding to `1 = Malignant` and `0 = Benign`.
3. Perform a stratified 75:25 train-test split.
4. Train all five classifiers using the 426 training instances.
5. Evaluate the models using the 143 held-out test instances.
6. Calculate Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
7. Save each fitted model pipeline as a `.joblib` file.
8. Export the held-out test partition as `test_data.csv`.
9. Load the saved models in Streamlit.
10. Evaluate uploaded test data without retraining the models.

---

## Run Locally

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Activate the environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Recreate the trained models and test data:

```bash
python train_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open the URL displayed by Streamlit, typically:

```text
http://localhost:8501
```

Upload `test_data.csv` and select any of the five trained classifiers.

---

## Streamlit Application Features

The application provides the following functionality:

- Upload labelled test data in CSV format
- Select any of the five trained classification models
- Load the saved `.joblib` model without retraining
- Display the number of evaluated test records
- Display Accuracy
- Display AUC Score
- Display Precision
- Display Recall
- Display F1 Score
- Display MCC Score
- Display a Confusion Matrix
- Display a Classification Report
- Validate required feature columns and the target column
- Download the sample `test_data.csv`

The uploaded CSV must contain all 30 required predictive features and the `target` column.


---

## Deployment on Streamlit Community Cloud

The application is deployed from this GitHub repository using Streamlit Community Cloud.

Deployment configuration:

```text
Repository: srikanthisin/ml-classification-streamlit
Branch: main
Main file path: app.py
```

Deployment steps:

1. Sign in to Streamlit Community Cloud using GitHub.
2. Create a new application.
3. Select the `srikanthisin/ml-classification-streamlit` repository.
4. Select the `main` branch.
5. Specify `app.py` as the application entry point.
6. Deploy the application.
7. Upload `test_data.csv` to the deployed application.
8. Verify evaluation using all five saved models.

---

## BITS Virtual Lab Execution

The project was executed and tested in the BITS Virtual Lab.

The following commands were used:

```bash
python train_models.py
streamlit run app.py
```

The BITS Virtual Lab execution verified:

- Training of all five classification models
- Generation of all five `.joblib` model files
- Generation of the 143-row `test_data.csv`
- Calculation of all six required evaluation metrics
- CSV upload functionality
- Model-selection functionality
- Evaluation without retraining
- Confusion matrix generation
- Classification report generation

A screenshot from the BITS Virtual Lab execution is included separately in the final assignment PDF as required.

---

## Reproducibility

The experiment uses:

```text
random_state = 42
test_size = 0.25
stratify = y
```

Running:

```bash
python train_models.py
```

recreates:

- Five trained `.joblib` model pipelines
- `model/schema.json`
- `test_data.csv`
- `results/model_metrics.csv`

The saved model files are subsequently loaded by `app.py`, so evaluation through the Streamlit application does not retrain any classifier.

---

## Final Assignment Submission

The final submitted PDF contains the following items in the required order:

1. Clickable GitHub repository link
2. Clickable live Streamlit application link
3. BITS Virtual Lab execution screenshot
4. Complete README content
