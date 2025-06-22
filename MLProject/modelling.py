import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import sys # Untuk menerima argumen dari MLProject

# --- MLflow Setup (Local for CI build, artifacts saved to mlruns/ in repo) ---
# MLflow will save artifacts to ./mlruns/ relative to where mlflow run is executed
    
features_path = sys.argv[4] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MLProject/telco_customer_churn_preprocessed_X.csv"
)
target_path = sys.argv[4] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "MLProject/telco_customer_churn_preprocessed_y.csv"
)
    
print(f"Reading features from: {features_path}")
print(f"Reading target from: {target_path}")
    
# Load preprocessed data (assuming these files are in MLproject/ folder)
# Adjust path if necessary
X_processed = pd.read_csv(features_path)
y_encoded = pd.read_csv(target_path)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- Model Training (Use best params from Kriteria 2) ---
# You can get these from your best run in Kriteria 2's MLflow UI
# Example best params from Logistic Regression: C=1.0, solver='liblinear'
# Example best params from RandomForest: n_estimators=200, max_depth=10

# Choose one model type and its best parameters
model_type = "LogisticRegression" # or "RandomForestClassifier"
if model_type == "LogisticRegression":
    model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000, class_weight='balanced')
    params = {"C": 1.0, "solver": "liblinear"}
else: # RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    params = {"n_estimators": 100, "max_depth": 10}

print(f"Training {model_type} with params: {params}")

with mlflow.start_run(run_name=f"CI_Run_{model_type}"):
    # Manual Logging: Parameters
    for key, value in params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("model_type", model_type)

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Manual Logging: Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the model artifact
    signature = mlflow.models.signature.infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        model, 
        "model", 
        signature=signature, 
        input_example=X_test.iloc[:5]
    )

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    plt.savefig("conf_matrix.png")

    # Untuk MLflow (optional tapi disarankan)
    mlflow.log_artifact("conf_matrix.png")
    
    # Log input_example
    input_example = X_train.iloc[:5]
    mlflow.log_input(mlflow.data.from_pandas(input_example), context="input_example")

    print(f"Training completed. Accuracy: {accuracy:.4f}")
