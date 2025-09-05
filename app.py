from fastapi import FastAPI, UploadFile, File
import pandas as pd
import mlflow.sklearn

app = FastAPI()

# Load latest model from MLflow
MODEL_URI = "models:/CreditCardFraudModel/1"  # v1 registered model
model = mlflow.sklearn.load_model(MODEL_URI)

@app.get("/")
def home():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded CSV
    contents = await file.read()
    df = pd.read_csv(pd.compat.StringIO(contents.decode("utf-8")))

    if "Class" in df.columns:
        df = df.drop("Class", axis=1)

    preds = model.predict(df)
    df["prediction"] = preds

    fraud_count = int((df["prediction"] == 1).sum())
    normal_count = int((df["prediction"] == 0).sum())

    return {
        "total_records": len(df),
        "fraudulent": fraud_count,
        "normal": normal_count
    }
