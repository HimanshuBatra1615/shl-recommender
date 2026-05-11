FROM python:3.11-slim

WORKDIR /app

# Install system deps for chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/
COPY data/ ./data/

# Pre-download default ONNX model to avoid download on startup
RUN python -c "from chromadb.utils import embedding_functions; embedding_functions.DefaultEmbeddingFunction()(['test'])"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
