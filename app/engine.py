import os
import json
import re

# Must be set before any FAISS/numpy import to prevent OpenMP crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple


CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"
VECTOR_STORE_PATH = Path(__file__).parent.parent / "data" / "vector_store"
ST_MODEL_PATH = Path(__file__).parent.parent / "data" / "st_model"

_SENTINEL = object()

# ---------------------------------------------------------------------------
# Seniority detection + job level mapping
# ---------------------------------------------------------------------------

_SENIORITY_KEYWORDS: Set[str] = {
    "senior", "lead", "principal", "staff", "director", "cxo", "vp", "c-suite",
    "c-level", "expert", "advanced", "experienced", "10+", "15+", "20+",
    "head of", "vp of", "chief", "executive", "manager", "supervisor",
    "front line manager",
}

# Maps user-facing seniority language → SHL catalog job_level values
_JOB_LEVEL_MAP: Dict[str, str] = {
    "entry":        "Entry-Level",
    "entry-level":  "Entry-Level",
    "junior":       "Entry-Level",
    "intern":       "Entry-Level",
    "graduate":     "Graduate",
    "grad":         "Graduate",
    "mid":          "Mid-Professional",
    "mid-level":    "Mid-Professional",
    "professional": "Mid-Professional",
    "supervisor":   "Supervisor",
    "manager":      "Manager",
    "front line":   "Front Line Manager",
    "frontline":    "Front Line Manager",
    "director":     "Director",
    "executive":    "Executive",
    "cxo":          "Executive",
    "vp":           "Executive",
    "c-suite":      "Executive",
    "senior":       "Professional Individual Contributor",
    "lead":         "Professional Individual Contributor",
    "principal":    "Professional Individual Contributor",
    "staff":        "Professional Individual Contributor",
}

# ---------------------------------------------------------------------------
# Framework → Catalog name mapping
# ---------------------------------------------------------------------------

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
}


def detect_seniority(text: str) -> bool:
    """Return True if any seniority signal is found in text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _SENIORITY_KEYWORDS)


def detect_job_level(text: str) -> Optional[str]:
    """Return the SHL catalog job_level string most closely matching the user's query, or None."""
    text_lower = text.lower()
    # Longest-match first to avoid "manager" matching "front line manager" substrings
    for kw in sorted(_JOB_LEVEL_MAP, key=len, reverse=True):
        if kw in text_lower:
            return _JOB_LEVEL_MAP[kw]
    return None


def extract_frameworks(text: str) -> List[str]:
    """Return list of framework keywords found in text (matched against FRAMEWORK_CATALOG_MAP)."""
    text_lower = text.lower()
    found = []
    for kw in FRAMEWORK_CATALOG_MAP:
        if kw in text_lower and kw not in found:
            found.append(kw)
    return found


def clean_duration(raw: str) -> str:
    """Normalise duration strings — return 'Variable' for em-dash/zero/unknown values."""
    if not raw or raw.strip() in ("", "-", "—", "0 minutes", "0", "?"):
        return "Variable"
    # Strip replacement character that appears when scraper hit an em-dash
    cleaned = raw.replace("�", "").strip()
    return cleaned if cleaned else "Variable"


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
        self._url_map: Dict[str, Dict] = {}
        self._model: Any = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load(self) -> None:
        raw_catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

        # PRD scope: only Individual Test Solutions are recommendable. SHL labels
        # its Pre-packaged Job Solutions with a trailing " Solution" in the name
        # (e.g. "Entry Level Sales Solution"); these bundle multiple test types
        # and are strictly out of scope. Filter at load time so the raw scraped
        # catalog.json stays intact but these are never retrievable/verifiable.
        self._catalog = [
            e for e in raw_catalog
            if not e["name"].strip().endswith(" Solution")
        ]
        dropped = len(raw_catalog) - len(self._catalog)

        self._name_map = {e["name"]: e for e in self._catalog}
        self._url_map = {e["url"].rstrip("/"): e for e in self._catalog}

        index_file = VECTOR_STORE_PATH / "catalog.index"
        ids_file   = VECTOR_STORE_PATH / "index_ids.json"

        if index_file.exists() and ids_file.exists():
            self._index = faiss.read_index(str(index_file))
            self._id_map = json.loads(ids_file.read_text(encoding="utf-8"))
        else:
            self._index = None

        print(f"[CatalogEngine] Loaded {len(self._catalog)} entries "
              f"({dropped} pre-packaged Job Solutions filtered out). "
              f"FAISS: {'ready' if self._index is not None else 'fallback'}.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            model_source = str(ST_MODEL_PATH) if ST_MODEL_PATH.exists() else "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(model_source)

    def _embed(self, text: str) -> np.ndarray:
        self._ensure_model()
        vec = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)

    def _keyword_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        tokens = set(re.sub(r"[^a-z0-9 ]", " ", query.lower()).split())
        tokens = {t for t in tokens if len(t) > 2}  # skip stopword-length tokens
        scored = []
        for entry in self._catalog:
            blob = (
                entry["name"] + " " +
                entry.get("description", "") + " " +
                " ".join(entry.get("keys", [])) + " " +
                " ".join(entry.get("job_levels", []))
            ).lower()
            score = sum(1 for t in tokens if t in blob)
            scored.append((score, entry))
        scored.sort(key=lambda x: -x[0])
        top = [e for s, e in scored[:k] if s > 0]
        return top if top else []

    # ------------------------------------------------------------------
    # URL/Name verification
    # ------------------------------------------------------------------

    def verify_item(self, name: str, url: str) -> Optional[Dict[str, Any]]:
        """Cross-reference a recommendation against the catalog."""
        if name in self._name_map:
            return self._name_map[name]

        name_lower = name.lower()
        for catalog_name, entry in self._name_map.items():
            if catalog_name.lower() == name_lower:
                return entry

        url_norm = url.rstrip("/")
        if url_norm in self._url_map:
            return self._url_map[url_norm]

        candidates = []
        for catalog_name, entry in self._name_map.items():
            cn_lower = catalog_name.lower()
            if name_lower in cn_lower or cn_lower in name_lower:
                candidates.append((len(cn_lower), entry))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]

        return None

    # ------------------------------------------------------------------
    # Boosted search with hybrid retrieval + job_levels awareness
    # ------------------------------------------------------------------

    def search_boosted(
        self,
        query: str,
        seniority: bool = False,
        frameworks: Optional[List[str]] = None,
        job_level: Optional[str] = None,
        k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid semantic + keyword search with three promotion layers:
        - job_level boost: items whose job_levels include the detected level float up
        - seniority boost: "Advanced Level" items promoted when senior signals present
        - framework pinning: exact framework-named tests locked at position 0
        Returns up to k deduplicated items.
        """
        pool_size = min(k * 4, self._index.ntotal if self._index else len(self._catalog))

        # Hybrid: semantic + keyword merged via score fusion
        semantic = self._semantic_search(query, pool_size)
        keyword  = self._keyword_search(query, pool_size // 2)

        score_map: Dict[str, float] = {}
        item_map:  Dict[str, Dict]  = {}

        for rank, item in enumerate(semantic):
            score_map[item["name"]] = 1.0 - rank / max(len(semantic), 1)
            item_map[item["name"]]  = item

        keyword_names = {item["name"] for item in keyword}
        for name in list(score_map):
            if name in keyword_names:
                score_map[name] += 0.15   # hybrid co-occurrence bonus

        for rank, item in enumerate(keyword):
            if item["name"] not in score_map:
                score_map[item["name"]] = max(0.3 - rank * 0.01, 0.05)
                item_map[item["name"]]  = item

        # Layer 0: job_level boost — items explicitly targeting the detected level
        if job_level:
            for name, item in item_map.items():
                if job_level in item.get("job_levels", []):
                    score_map[name] += 0.20

        base_results = [
            item_map[name]
            for name in sorted(score_map, key=lambda n: -score_map[n])
        ]

        # Layer 1: seniority → prefer "Advanced Level" tests
        if seniority:
            advanced = [r for r in base_results
                        if re.search(r"advanced level|senior", r["name"], re.I)]
            others   = [r for r in base_results if r not in advanced]
            base_results = advanced + others

        # Layer 2: pin exact framework tests at the front
        pinned: List[Dict[str, Any]] = []
        if frameworks:
            seen_pinned: Set[str] = set()
            for fw in frameworks:
                for catalog_name in FRAMEWORK_CATALOG_MAP.get(fw, []):
                    item = self.get_by_name(catalog_name)
                    if item and item["name"] not in seen_pinned:
                        pinned.append(item)
                        seen_pinned.add(item["name"])

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
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        return self._semantic_search(query, k)

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self._name_map.get(name)

    def get_all(self) -> List[Dict[str, Any]]:
        return list(self._catalog)

    def format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        if not candidates:
            return "No catalog entries retrieved for this query."
        lines = []
        for e in candidates:
            duration = clean_duration(e.get("duration", ""))
            job_levels = ", ".join(e.get("job_levels", [])) or "All levels"
            lines.append(
                f"NAME: {e['name']}\n"
                f"  URL: {e['url']}\n"
                f"  TYPE: {e['test_type']} | {', '.join(e.get('keys', []))}\n"
                f"  DURATION: {duration}\n"
                f"  JOB LEVELS: {job_levels}\n"
                f"  LANGUAGES: {', '.join(e.get('languages', [])[:6])}\n"
                f"  DESCRIPTION: {e.get('description', '')}"
            )
        return "\n\n".join(lines)
