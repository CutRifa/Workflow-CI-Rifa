import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Load Data dari folder preprocessed
data_path = "stellar_preprocessed"
X_train = pd.read_csv(f"{data_path}/X_train.csv")
y_train = pd.read_csv(f"{data_path}/y_train.csv")
X_test = pd.read_csv(f"{data_path}/X_test.csv")
y_test = pd.read_csv(f"{data_path}/y_test.csv")

# 2. Setup MLflow
mlflow.set_experiment("Stellar_Classification_Project")

with mlflow.start_run():
    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    
    # Eval
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {acc}")
    
    # Log Artifacts
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model_bintang")