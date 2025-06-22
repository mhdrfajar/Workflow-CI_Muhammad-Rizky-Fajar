import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import os
import sys
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

if __name__ == "__main__":
    
    features_path = sys.argv[3] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_customer_churn_preprocessed_X.csv")
    target_path = sys.argv[4] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "telco_customer_churn_preprocessed_y.csv")
    
    X_processed = pd.read_csv(features_path)
    y_encoded = pd.read_csv(target_path)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # --- Model Training ---
    model_type = "LogisticRegression"  # or "RandomForestClassifier"
    if model_type == "LogisticRegression":
        model = LogisticRegression(C=1.0, solver='liblinear', random_state=42, max_iter=1000, class_weight='balanced')
        params = {"C": 1.0, "solver": "liblinear"}
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
        params = {"n_estimators": 100, "max_depth": 10}

    print(f"Training {model_type} with params: {params}")

    with mlflow.start_run(run_name=f"CI_Run_{model_type}"):
        # Log parameters
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

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log model artifact
        signature = mlflow.models.signature.infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            model, 
            "model", 
            signature=signature, 
            input_example=X_test.iloc[:5]
        )

        # Confusion matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        plt.savefig("conf_matrix.png")
        mlflow.log_artifact("conf_matrix.png")

        # Log input example
        input_example = X_train.iloc[:5]
        input_example.to_csv("input_example.csv", index=False)
        mlflow.log_artifact("input_example.csv")

        print(f"Training completed. Accuracy: {accuracy:.4f}")
