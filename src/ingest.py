"""
ingest.py
Parses Google Takeout CSV export and enriches each
restaurant with live data from the Google Places API.
"""

import csv
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def load_csv(filepath: str) -> list[dict]:
    """
    Load saved places from a Google Takeout CSV export.
    Expected columns: Title, Note, URL, Tags, Comment
    """
    places = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            places.append({
                "name": row.get("Title", "").strip(),
                "note": row.get("Note", "").strip(),
                "url": row.get("URL", "").strip(),
                "tags": [t.strip() for t in row.get("Tags", "").split(",") if t.strip()],
                "comment": row.get("Comment", "").strip(),
            })
    return places


def find_place(name: str) -> str | None:
    """
    Use Places API Text Search to find a place_id from a restaurant name.
    Returns the place_id string or None if not found.
    """
    endpoint = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "input": name,
        "inputtype": "textquery",
        "fields": "place_id",
        "key": PLACES_API_KEY,
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    candidates = data.get("candidates", [])
    if candidates:
        return candidates[0].get("place_id")
    return None


def get_place_details(place_id: str) -> dict:
    """
    Use Place Details API to fetch rich metadata for a place_id.
    Requests only fields in the Basic and Atmosphere categories
    to minimize billing.
    """
    endpoint = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": ",".join([
            "name",
            "formatted_address",
            "geometry/location",
            "types",
            "price_level",
            "rating",
            "business_status",
            "opening_hours/open_now",
            "opening_hours/weekday_text",
            "website",
        ]),
        "key": PLACES_API_KEY,
    }
    response = requests.get(endpoint, params=params)
    data = response.json()
    return data.get("result", {})


def enrich_place(place: dict) -> dict:
    """
    Find a place via Text Search, then enrich with Place Details.
    Returns the place dict with additional fields merged in.
    """
    place_id = find_place(place["name"])
    if not place_id:
        print(f"    WARNING: Could not find place_id for '{place['name']}'")
        return place

    details = get_place_details(place_id)

    location = details.get("geometry", {}).get("location", {})
    opening_hours = details.get("opening_hours", {})

    place["place_id"] = place_id
    place["address"] = details.get("formatted_address", "")
    place["latitude"] = location.get("lat")
    place["longitude"] = location.get("lng")
    place["types"] = details.get("types", [])
    place["price_level"] = details.get("price_level")  # 0-4
    place["rating"] = details.get("rating")
    place["business_status"] = details.get("business_status", "OPERATIONAL")  # OPERATIONAL, CLOSED_TEMPORARILY, CLOSED_PERMANENTLY
    place["open_now"] = opening_hours.get("open_now")
    place["hours"] = opening_hours.get("weekday_text", [])  # e.g. ["Monday: 9am-10pm", ...]
    place["website"] = details.get("website", "")

    return place


def build_document_text(place: dict) -> str:
    """
    Convert a place dict into a rich text string for embedding.
    The more detail here, the better the semantic search quality.
    """
    price_map = {0: "free", 1: "inexpensive", 2: "moderate", 3: "expensive", 4: "very expensive"}
    price_str = price_map.get(place.get("price_level"), "unknown")

    hours_str = "\n  ".join(place.get("hours", [])) or "unknown"

    business_status = place.get("business_status", "OPERATIONAL")

    parts = [
        f"Name: {place.get('name', '')}",
        f"Address: {place.get('address', '')}",
        f"Business status: {business_status}",
        f"Cuisine / types: {', '.join(place.get('types', []))}",
        f"Price level: {price_str}",
        f"Rating: {place.get('rating', 'unknown')} / 5",
        f"Open now: {place.get('open_now', 'unknown')}",
        f"Hours:\n  {hours_str}",
        f"Your note: {place.get('note') or place.get('comment') or 'none'}",
        f"Your tags: {', '.join(place.get('tags', [])) or 'none'}",
    ]
    return "\n".join(parts)


def run_ingestion(csv_filepath: str, delay: float = 0.1) -> list[dict]:
    """
    Full ingestion pipeline:
    1. Load CSV export
    2. Find each place via Text Search
    3. Enrich with Place Details
    4. Build document text for embedding
    Returns enriched place list ready for embedding.

    delay: seconds to wait between API calls (be nice to the API)
    """
    raw_places = load_csv(csv_filepath)
    print(f"Loaded {len(raw_places)} saved places from CSV.")

    enriched = []
    for i, place in enumerate(raw_places):
        print(f"  [{i+1}/{len(raw_places)}] Processing: {place['name']}")
        place = enrich_place(place)
        place["document_text"] = build_document_text(place)
        enriched.append(place)
        time.sleep(delay)  # avoid hammering the API

    succeeded = sum(1 for p in enriched if p.get("place_id"))
    print(f"\nIngestion complete. {succeeded}/{len(enriched)} places successfully enriched.")
    return enriched


if __name__ == "__main__":
    import json
    places = run_ingestion("data/raw/Saved Places.csv")
    # Print first result as a sanity check
    if places:
        print("\nSample output:")
        sample = {k: v for k, v in places[0].items() if k != "embedding"}
        print(json.dumps(sample, indent=2))
