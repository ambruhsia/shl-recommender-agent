FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .

# Install CPU-only PyTorch first to avoid pulling CUDA/GPU variants (~2 GB of GPU libs).
# The CPU wheel (~200 MB) is sufficient for inference on any CPU-only host.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (torch already satisfied, no GPU libs pulled)
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app/ app/
COPY --chown=user scripts/ scripts/
COPY --chown=user data/catalog.json data/catalog.json

# Build FAISS index + save sentence-transformers model to data/st_model/
# Both are baked into the image — zero network calls at runtime
RUN KMP_DUPLICATE_LIB_OK=TRUE python scripts/build_catalog.py

ENV KMP_DUPLICATE_LIB_OK=TRUE

# HF Spaces requires port 7860; PORT env var lets Render/other hosts override
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
