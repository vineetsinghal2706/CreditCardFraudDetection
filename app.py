from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]  # uploaded CSV
    data = pd.read_csv(file)

    if "Class" in data.columns:  # drop if present
        data = data.drop("Class", axis=1)

    preds = model.predict(data)
    data["prediction"] = preds

    # Return sample of predictions
    return jsonify({
        "rows": len(data),
        "sample_predictions": data.head(10).to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
