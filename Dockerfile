FROM python:3.12.5-slim

# libgomp1 is required by PyTorch/FAISS for OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies as a separate layer so it's cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and catalog
COPY app/ app/
COPY scripts/ scripts/
COPY data/catalog.json data/catalog.json

# Build FAISS index + save sentence-transformers model to data/st_model/
# Both are baked into the image — zero network calls at runtime
RUN KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_catalog.py

ENV KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 10000

# PORT is injected by Render at runtime; default to 10000 for local docker run
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
