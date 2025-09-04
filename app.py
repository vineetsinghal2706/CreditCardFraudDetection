from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return {"message": "Credit Card Fraud Detection API"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  
    prediction = model.predict([np.array(data)])
    return jsonify({"fraud_prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
