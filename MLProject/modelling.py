import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Load Data ---
# Pastikan path ini sesuai dengan struktur folder di CI nanti
# Kita gunakan try-except agar aman dijalankan di berbagai environment
try:
    X_train = pd.read_csv("wine_preprocessed/X_train.csv")
    y_train = pd.read_csv("wine_preprocessed/y_train.csv").values.ravel()
    X_test = pd.read_csv("wine_preprocessed/X_test.csv")
    y_test = pd.read_csv("wine_preprocessed/y_test.csv").values.ravel()
except FileNotFoundError:
    # Fallback untuk Kriteria 3 jika struktur folder sedikit berbeda
    X_train = pd.read_csv("namadataset_preprocessing/X_train.csv")
    y_train = pd.read_csv("namadataset_preprocessing/y_train.csv").values.ravel()
    X_test = pd.read_csv("namadataset_preprocessing/X_test.csv")
    y_test = pd.read_csv("namadataset_preprocessing/y_test.csv").values.ravel()

# --- MLflow Run ---
mlflow.set_experiment("Eksperimen_Wine_Simple")

with mlflow.start_run(run_name="Basic_RandomForest"):
    # Definisi Model Sederhana
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc}")
    
    # Logging Sederhana
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")