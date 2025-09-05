FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and trained model
COPY app.py .
COPY model.pkl .

EXPOSE 5000
CMD ["python", "app.py"]
