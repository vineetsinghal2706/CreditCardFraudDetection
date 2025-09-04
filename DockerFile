FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py app.py creditcard.csv ./

# Train the model when building
RUN python train.py

EXPOSE 8000

CMD ["python", "app.py"]
