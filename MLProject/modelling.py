import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load Data Iris
print("[INFO] Loading data Iris...")
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
y = iris.target

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow
mlflow.set_experiment("Eksperimen_Iris_Rifa_CI")

# 3. Training
with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Prediksi
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")
    
    # Logging
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model_iris") # Nama folder model
    
    print("[SUCCESS] Training Selesai.")