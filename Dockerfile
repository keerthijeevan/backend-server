FROM python:3.12-slim

WORKDIR /app

COPY requiremnts.txt .

RUN pip install --no-cache-dir -r requiremnts.txt

COPY app.py .

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
