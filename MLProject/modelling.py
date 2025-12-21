import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- 1. Load Data ---
# PERBAIKAN: Menggunakan nama folder 'iris_preprocessing' (sesuai folder Anda)
folder_name = "iris_preprocessing" 
print(f"Memuat data dari folder '{folder_name}'...")

try:
    X_train = pd.read_csv(f"{folder_name}/X_train.csv")
    y_train = pd.read_csv(f"{folder_name}/y_train.csv").values.ravel()
    X_test = pd.read_csv(f"{folder_name}/X_test.csv")
    y_test = pd.read_csv(f"{folder_name}/y_test.csv").values.ravel()
    
    print("✅ Data berhasil dimuat!")

except FileNotFoundError:
    print(f"❌ Error: Folder '{folder_name}' tidak ditemukan.")
    print("Cek nama folder Anda. Apakah 'iris_preprocessing' atau 'iris_preprocessed'?")
    print(f"Lokasi saat ini: {os.getcwd()}")
    exit()

# --- 2. Setup MLflow ---
mlflow.set_experiment("Eksperimen_Iris_Final")

# --- 3. Training ---
print("Mulai Training...")

with mlflow.start_run(run_name="Iris_RandomForest"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model_random_forest")
    
    print("✅ Model tersimpan di MLflow.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")