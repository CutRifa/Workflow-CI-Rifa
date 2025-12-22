import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- KRITERIA 2: Aktifkan Logging Otomatis ---
mlflow.autolog()

# 1. Load Data dari folder preprocessed
data_path = "stellar_preprocessed"
X_train = pd.read_csv(f"{data_path}/X_train.csv")
y_train = pd.read_csv(f"{data_path}/y_train.csv")
X_test = pd.read_csv(f"{data_path}/X_test.csv")
y_test = pd.read_csv(f"{data_path}/y_test.csv")

# 3. Training
# Parameter dan metrik akan dicatat otomatis oleh autolog
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train.values.ravel())

# 4. Evaluation
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc}")