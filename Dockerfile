FROM python:3.11.9-slim

WORKDIR /app

# Cài các công cụ build và thư viện cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy code
COPY requirements.txt .
COPY main.py .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Chạy FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]