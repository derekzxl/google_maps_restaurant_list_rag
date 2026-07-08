"""
populate_qdrant.py
One-time script to ingest both CSV files, enrich with Places API,
embed with Gemini, and index in Qdrant.
Run this once before starting the Flask app.
Usage: python scripts/populate_qdrant.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import load_csv, enrich_place, build_document_text
from src.embed import embed_places
from src.index import get_client, create_collection, upsert_places
import time

LISTS = [
    {"filepath": "data/raw/food.csv", "list_name": "visited"},
    {"filepath": "data/raw/want to go.csv", "list_name": "want_to_go"},
]


def run_ingestion_for_list(filepath: str, list_name: str, delay: float = 0.1) -> list[dict]:
    places = load_csv(filepath)
    print(f"  Loaded {len(places)} places from {filepath}")

    enriched = []
    for i, place in enumerate(places):
        print(f"    [{i+1}/{len(places)}] Processing: {place['name']}")
        place = enrich_place(place)
        place["list_name"] = list_name  # tag which list this came from
        place["document_text"] = build_document_text(place)
        enriched.append(place)
        time.sleep(delay)

    # Filter out failed enrichments
    failed = [p for p in enriched if not p.get("place_id")]
    if failed:
        print(f"\n  ⚠️ Skipping {len(failed)} places that failed Place Details enrichment:")
        for p in failed:
            print(f"    - {p['name']}")
    enriched = [p for p in enriched if p.get("place_id")]

    return enriched


def main():
    import json
    interim_dir = "data/interim"
    os.makedirs(interim_dir, exist_ok=True)
    cache_file = os.path.join(interim_dir, "enriched_places.json")

    all_places = []
    
    if os.path.exists(cache_file):
        print(f"\n=== Loading enriched places from cache: {cache_file} ===")
        with open(cache_file, "r", encoding="utf-8") as f:
            all_places = json.load(f)
    else:
        for lst in LISTS:
            print(f"\n=== Ingesting '{lst['list_name']}' from {lst['filepath']} ===")
            places = run_ingestion_for_list(lst["filepath"], lst["list_name"])
            all_places.extend(places)
        
        print(f"\n=== Saving {len(all_places)} enriched places to cache ===")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(all_places, f, indent=2)

    print(f"\n=== Total places to embed: {len(all_places)} ===")

    print("\n=== Generating embeddings ===")
    all_places = embed_places(all_places)

    print("\n=== Indexing in Qdrant ===")
    client = get_client()
    create_collection(client)
    upsert_places(client, all_places)

    print("\n✓ Index built successfully. You can now run the Flask app.")


if __name__ == "__main__":
    main()
