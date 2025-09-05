import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

# -------------------------
# Load first 150,000 rows
# -------------------------
data = pd.read_csv("creditcard.csv", nrows=150000)

if "Class" not in data.columns:
    raise ValueError(f"'Class' column not found. Available columns: {list(data.columns)}")

# -------------------------
# Convert feature columns to float64 to avoid MLflow warnings
# -------------------------
X = data.drop("Class", axis=1).astype("float64")
y = data["Class"]

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# Train model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Predictions & metrics
# -------------------------
y_pred = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred)
}

print("✅ Model Trained")
for k, v in metrics.items():
    print(f"{k.capitalize()}: {v:.4f}")

# -------------------------
# MLflo
