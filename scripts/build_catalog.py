"""
Build the FAISS vector index from data/catalog.json.
Run once before starting the server:
    python scripts/build_catalog.py
"""
import os
import json
import sys
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Resolve paths relative to repo root (this script lives in scripts/)
REPO_ROOT = Path(__file__).parent.parent
CATALOG_PATH = REPO_ROOT / "data" / "catalog.json"
VECTOR_STORE_PATH = REPO_ROOT / "data" / "vector_store"


def build_text(entry: dict) -> str:
    langs = ", ".join(entry.get("languages", [])[:4])
    keys = ", ".join(entry.get("keys", []))
    return (
        f"{entry['name']}. {entry.get('description', '')} "
        f"Type: {keys}. Duration: {entry.get('duration', 'unknown')}. "
        f"Languages: {langs}."
    )


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"ERROR: catalog.json not found at {CATALOG_PATH}", file=sys.stderr)
        sys.exit(1)

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(catalog)} catalog entries.")

    texts = [build_text(e) for e in catalog]
    names = [e["name"] for e in catalog]

    print("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding catalog entries…")
    embeddings = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )
    embeddings = embeddings.astype(np.float32)

    dim = embeddings.shape[1]
    print(f"Embedding dimension: {dim}")

    import faiss
    index = faiss.IndexFlatIP(dim)   # inner product on L2-normalised = cosine similarity
    index.add(embeddings)

    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
    index_path = VECTOR_STORE_PATH / "catalog.index"
    ids_path = VECTOR_STORE_PATH / "index_ids.json"

    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved to: {index_path}")
    print(f"ID map saved to: {ids_path}")


if __name__ == "__main__":
    main()
