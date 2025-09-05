FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py fraud_model.pkl ./

EXPOSE 5000

CMD ["python", "app.py"]
