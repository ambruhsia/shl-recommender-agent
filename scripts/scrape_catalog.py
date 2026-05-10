"""
Fetch and process the SHL product catalog.

Two modes:
  1. Download raw catalog from the official JSON endpoint, parse and transform
     each entry into our schema, save to data/catalog.json.
  2. If individual pages need details not present in the JSON, fall back to
     BeautifulSoup scraping of each product URL.

Usage:
    python scripts/scrape_catalog.py
"""
import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
REPO_ROOT   = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "data" / "catalog.json"

# Map catalog key labels → single-letter type codes
KEY_TO_TYPE = {
    "Personality & Behavior":       "P",
    "Ability & Aptitude":           "A",
    "Knowledge & Skills":           "K",
    "Simulations":                  "S",
    "Biodata & Situational Judgment": "B",
    "Competencies":                 "C",
    "Development & 360":            "D",
    "Assessment Exercises":         "E",
}


def derive_type_code(keys: list[str]) -> str:
    """Convert a list of key labels into a comma-separated type-code string."""
    codes = []
    for k in keys:
        code = KEY_TO_TYPE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


def clean_text(text: str) -> str:
    """Remove embedded control characters that break JSON parsing."""
    # Replace control chars except \t \n \r with a space
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text).strip()


def fetch_raw_catalog() -> list[dict]:
    """Download and parse the raw catalog JSON from the endpoint."""
    print(f"Fetching catalog from {CATALOG_URL} …")
    resp = requests.get(CATALOG_URL, timeout=30)
    resp.raise_for_status()

    # Decode as UTF-8, replace any undecodable bytes
    raw = resp.content.decode("utf-8", errors="replace")

    # Strip embedded control characters before parsing
    raw = clean_text(raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Last resort: use strict=False to tolerate control chars in strings
        data = json.loads(raw, strict=False)

    if isinstance(data, dict):
        # Some endpoints wrap the array in a key
        for key in ("products", "catalog", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                return data[key]
        # Fall back to first list value
        for v in data.values():
            if isinstance(v, list):
                return v
        return [data]

    return data  # already a list


def scrape_product_page(url: str) -> dict:
    """
    BeautifulSoup fallback: scrape a single SHL product page to extract
    name, description, test_type, duration, and languages.
    Returns a partial dict (caller should merge with catalog entry).
    """
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as exc:
        print(f"  WARNING: could not fetch {url}: {exc}")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract description from meta or page body
    description = ""
    meta_desc = soup.find("meta", {"name": "description"}) or soup.find(
        "meta", {"property": "og:description"}
    )
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    if not description:
        # Try common SHL page selectors
        for selector in [".product-description", ".catalog-description", "article p"]:
            el = soup.select_one(selector)
            if el:
                description = el.get_text(separator=" ").strip()
                break

    # Extract duration
    duration = ""
    for tag in soup.find_all(string=re.compile(r"minute", re.I)):
        m = re.search(r"(\d+)\s*minute", str(tag), re.I)
        if m:
            duration = f"{m.group(1)} minutes"
            break

    return {"description": description, "duration": duration}


def transform_entry(raw: dict) -> dict | None:
    """Convert a raw catalog entry into our schema. Returns None to skip."""
    name = (raw.get("name") or "").strip()
    url  = raw.get("link") or raw.get("url") or ""
    if not name or not url:
        return None

    keys        = raw.get("keys") or []
    description = clean_text(raw.get("description") or "")
    duration    = (raw.get("duration") or raw.get("duration_raw") or "").strip()
    languages   = raw.get("languages") or []
    job_levels  = raw.get("job_levels") or []

    # Normalise duration: prefer "X minutes" format
    m = re.search(r"(\d+)", duration)
    if m and "minute" not in duration.lower():
        duration = f"{m.group(1)} minutes"

    test_type = derive_type_code(keys)

    entry = {
        "name":        name,
        "url":         url.strip(),
        "test_type":   test_type,
        "keys":        keys,
        "duration":    duration if duration else "—",
        "languages":   [str(l) for l in languages],
        "job_levels":  [str(j) for j in job_levels],
        "description": description,
        "remote":      raw.get("remote", ""),
        "adaptive":    raw.get("adaptive", ""),
    }
    return entry


def main() -> None:
    # Step 1 — Download and parse raw catalog
    raw_entries = fetch_raw_catalog()
    print(f"Raw catalog has {len(raw_entries)} entries.")

    # Step 2 — Transform each entry
    catalog: list[dict] = []
    missing_desc: list[dict] = []

    for raw in raw_entries:
        entry = transform_entry(raw)
        if entry is None:
            continue
        catalog.append(entry)
        if not entry["description"]:
            missing_desc.append(entry)

    print(f"Transformed {len(catalog)} entries.")
    print(f"Entries missing description: {len(missing_desc)}")

    # Step 3 — BeautifulSoup fallback for entries with no description
    if missing_desc:
        print(f"Scraping {len(missing_desc)} product pages for descriptions…")
        for i, entry in enumerate(missing_desc, 1):
            print(f"  [{i}/{len(missing_desc)}] {entry['name']}")
            scraped = scrape_product_page(entry["url"])
            if scraped.get("description"):
                entry["description"] = scraped["description"]
            if scraped.get("duration") and entry["duration"] == "—":
                entry["duration"] = scraped["duration"]
            time.sleep(0.3)  # be polite

    # Step 4 — Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {len(catalog)} entries to {OUTPUT_PATH}")

    # Summary
    type_counts: dict[str, int] = {}
    for e in catalog:
        type_counts[e["test_type"]] = type_counts.get(e["test_type"], 0) + 1
    print("Type distribution:", dict(sorted(type_counts.items())))


if __name__ == "__main__":
    main()
