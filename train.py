import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("creditcard.csv")

# Use 150k rows for training
train_data = data.iloc[:150000]

X = train_data.drop("Class", axis=1)
y = train_data["Class"]

# Train/test split (80/20 inside 150k subset)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Save model
joblib.dump(model, "fraud_model.pkl")
print("✅ Model saved as fraud_model.pkl")
