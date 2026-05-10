import os
import json
import re

# Must be set before any FAISS/numpy import to prevent OpenMP crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store"

_SENTINEL = object()

# ---------------------------------------------------------------------------
# Directive 2 & 3: Seniority detection + Framework → Catalog name mapping
# ---------------------------------------------------------------------------

_SENIORITY_KEYWORDS: Set[str] = {
    "senior", "lead", "principal", "staff", "director", "cxo", "vp", "c-suite",
    "c-level", "expert", "advanced", "experienced", "10+", "15+", "20+",
    "head of", "vp of", "chief", "executive",
}

# Maps lowercase technology keyword → list of exact catalog product names (priority order)
# Key insight: specific framework names → exact test names from the real catalog
FRAMEWORK_CATALOG_MAP: Dict[str, List[str]] = {
    "java":          ["Core Java (Advanced Level) (New)", "Core Java (Entry Level) (New)"],
    "spring":        ["Spring (New)"],
    "springboot":    ["Spring (New)"],
    "spring boot":   ["Spring (New)"],
    "sql":           ["SQL (New)"],
    "mysql":         ["SQL (New)"],
    "postgresql":    ["SQL (New)"],
    "docker":        ["Docker (New)"],
    "kubernetes":    ["Kubernetes (New)", "Docker (New)"],
    "aws":           ["Amazon Web Services (AWS) Development (New)"],
    "amazon web":    ["Amazon Web Services (AWS) Development (New)"],
    "rest":          ["RESTful Web Services (New)"],
    "restful":       ["RESTful Web Services (New)"],
    "linux":         ["Linux Programming (General)"],
    "networking":    ["Networking and Implementation (New)"],
    "python":        ["Python (New)"],
    "javascript":    ["JavaScript (New)"],
    "angular":       ["Angular (New)"],
    "react":         ["React (New)"],
    "c#":            [".NET Framework (New)", "C# (New)"],
    ".net":          [".NET Framework (New)"],
    "excel":         ["Microsoft Excel 365 (New)", "MS Excel (New)"],
    "word":          ["Microsoft Word 365 (New)", "MS Word (New)"],
    "hipaa":         ["HIPAA (Security)"],
    "salesforce":    ["Salesforce (New)"],
    "power bi":      ["Power BI (New)"],
    "tableau":       ["Tableau (New)"],
    # no Rust-specific test exists — handle separately in graph.py
}


def detect_seniority(text: str) -> bool:
    """Return True if any seniority signal is found in text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _SENIORITY_KEYWORDS)


def extract_frameworks(text: str) -> List[str]:
    """Return list of framework keywords found in text (matched against FRAMEWORK_CATALOG_MAP)."""
    text_lower = text.lower()
    found = []
    for kw in FRAMEWORK_CATALOG_MAP:
        if kw in text_lower and kw not in found:
            found.append(kw)
    return found


class CatalogEngine:
    """
    Holds the SHL product catalog, FAISS index, and sentence-transformer model.
    Call .load() once at application startup via FastAPI lifespan.
    """

    def __init__(self) -> None:
        self._catalog: List[Dict[str, Any]] = []
        self._index: Any = _SENTINEL
        self._id_map: List[str] = []
        self._name_map: Dict[str, Dict] = {}
        self._url_map: Dict[str, Dict] = {}      # url (normalised) → entry
        self._model: Any = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        self._catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self._name_map = {e["name"]: e for e in self._catalog}
        self._url_map = {e["url"].rstrip("/"): e for e in self._catalog}

        index_file = VECTOR_STORE_PATH / "catalog.index"
        ids_file = VECTOR_STORE_PATH / "index_ids.json"

        if index_file.exists() and ids_file.exists():
            self._index = faiss.read_index(str(index_file))
            self._id_map = json.loads(ids_file.read_text(encoding="utf-8"))
        else:
            self._index = None

        print(f"[CatalogEngine] Loaded {len(self._catalog)} entries. "
              f"FAISS: {'ready' if self._index is not None else 'fallback'}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        vec = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
        scored = []
        for entry in self._catalog:
            blob = (
                entry["name"] + " " +
                entry.get("description", "") + " " +
                " ".join(entry.get("keys", []))
            ).lower()
            score = sum(1 for t in tokens if t in blob)
            scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        top = [e for s, e in scored[:k] if s > 0]
        return top if top else self._catalog[:k]

    # ------------------------------------------------------------------
    # Directive 8: URL/Name verification
    # ------------------------------------------------------------------

    def verify_item(self, name: str, url: str) -> Optional[Dict[str, Any]]:
        """
        Cross-reference a recommendation against the catalog.
        Priority order:
          1. Exact name match
          2. Case-insensitive name match
          3. URL match (normalised)
          4. Partial name containment (longest match wins)
        Returns the canonical catalog entry, or None if hallucinated.
        """
        # 1. Exact name
        if name in self._name_map:
            return self._name_map[name]

        # 2. Case-insensitive name
        name_lower = name.lower()
        for catalog_name, entry in self._name_map.items():
            if catalog_name.lower() == name_lower:
                return entry

        # 3. URL match
        url_norm = url.rstrip("/")
        if url_norm in self._url_map:
            return self._url_map[url_norm]

        # 4. Partial name containment — pick the catalog entry whose name is most similar
        candidates = []
        for catalog_name, entry in self._name_map.items():
            cn_lower = catalog_name.lower()
            if name_lower in cn_lower or cn_lower in name_lower:
                candidates.append((len(cn_lower), entry))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]  # shortest (most specific) match

        return None  # hallucinated — will be dropped

    # ------------------------------------------------------------------
    # Directive 2 & 3: Boosted search
    # ------------------------------------------------------------------

    def search_boosted(
        self,
        query: str,
        seniority: bool = False,
        frameworks: Optional[List[str]] = None,
        k: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with two recall-boosting layers:
        - Directive 2: if seniority=True, promote "Advanced Level" items to the front
        - Directive 3: for each detected framework, pin the exact named test at position 0
        Returns up to k items, deduplicated.
        """
        # Fetch a larger pool so we have items to promote from
        pool_size = min(k * 2, self._index.ntotal if self._index else len(self._catalog))
        base_results = self._semantic_search(query, pool_size)

        # Layer 1 — Directive 2: seniority bias
        if seniority:
            advanced = [r for r in base_results
                        if re.search(r"advanced level|senior", r["name"], re.I)]
            others = [r for r in base_results if r not in advanced]
            base_results = advanced + others

        # Layer 2 — Directive 3: pin exact framework tests at the front
        pinned: List[Dict[str, Any]] = []
        if frameworks:
            seen_pinned: Set[str] = set()
            for fw in frameworks:
                for catalog_name in FRAMEWORK_CATALOG_MAP.get(fw, []):
                    item = self.get_by_name(catalog_name)
                    if item and item["name"] not in seen_pinned:
                        pinned.append(item)
                        seen_pinned.add(item["name"])

        # Merge: pinned first, then semantic results, deduplicated
        merged: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()
        for item in pinned + base_results:
            if item["name"] not in seen_names:
                merged.append(item)
                seen_names.add(item["name"])

        return merged[:k]

    def _semantic_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        if self._index is None:
            return self._keyword_search(query, k)
        vec = self._embed(query)
        n = min(k, self._index.ntotal)
        distances, indices = self._index.search(vec, n)
        results = []
        for idx in indices[0]:
            if idx < 0:
                continue
            name = self._id_map[idx]
            if name in self._name_map:
                results.append(self._name_map[name])
        return results

    # ------------------------------------------------------------------
    # Public API (original search kept for compatibility)
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 15) -> List[Dict[str, Any]]:
        return self._semantic_search(query, k)

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self._name_map.get(name)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._catalog)

    def catalog_summary_text(self) -> str:
        lines = []
        for e in self._catalog:
            keys = ", ".join(e.get("keys", []))
            langs = ", ".join(e.get("languages", [])[:3])
            lines.append(
                f"- {e['name']} | Type: {e['test_type']} ({keys}) | "
                f"Duration: {e.get('duration', '?')} | Lang: {langs}"
            )
        return "\n".join(lines)

    def format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        if not candidates:
            return "No catalog entries retrieved for this query."
        lines = []
        for e in candidates:
            lines.append(
                f"NAME: {e['name']}\n"
                f"  URL: {e['url']}\n"
                f"  TYPE: {e['test_type']} | {', '.join(e.get('keys', []))}\n"
                f"  DURATION: {e.get('duration', '?')}\n"
                f"  LANGUAGES: {', '.join(e.get('languages', [])[:6])}\n"
                f"  DESCRIPTION: {e.get('description', '')}"
            )
        return "\n\n".join(lines)
