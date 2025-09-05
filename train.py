import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

import os
import sys


file_path = "creditcard.csv"

if not os.path.exists(file_path):
    print(f"❌ Dataset not found: {file_path}")
    sys.exit(1)

data = pd.read_csv(file_path)

if "Class" not in data.columns:
    print(f"❌ 'Class' column not found in dataset. Available columns: {list(data.columns)}")
    sys.exit(1)

# Load dataset
data = pd.read_csv("creditcard.csv")

# Use 150k rows for training
train_data = data.iloc[:150000]

# Prepare features and labels
X = data.drop(['Class'], axis=1)
y = data['Class']

# Train-test split
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Save model
joblib.dump(rfc, "fraud_model.pkl")
print("✅ Model trained and saved as fraud_model.pkl")
