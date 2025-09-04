import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Load trained model
with open("fraud_model.pkl", "rb") as f:
    model = joblib.load(f)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "✅ Credit Card Fraud Detection API is running!"}

@app.route("/predict", methods=["POST"])
def predict():
    # Check if a file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        data = pd.read_csv(file)

        # Drop target column if present
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        # Predict
        preds = model.predict(data)

        # Return JSON response
        return jsonify({"predictions": preds.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
