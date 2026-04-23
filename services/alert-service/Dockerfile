FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Run consumer in background, expose FastAPI status API
CMD ["sh", "-c", "python consumer.py & uvicorn main:app --host 0.0.0.0 --port 8005"]
