FROM python:3.11.9

WORKDIR /app

# Cài các công cụ build và thư viện cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    python3-dev \
    gcc \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Cập nhật pip
RUN pip install --upgrade pip

# Copy code
COPY requirements.txt .
COPY main.py .

# Cài lightfm trước
RUN pip install lightfm==1.17

# Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt

# Chạy FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT