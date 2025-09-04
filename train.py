import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

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
