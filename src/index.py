"""
index.py
Handles storing and retrieving place vectors from Qdrant Cloud.
Supports optional metadata filtering by list_name (visited / want_to_go).
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "saved_places"
VECTOR_SIZE = 3072  # gemini-embedding-001 default output dimension


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


def create_collection(client: QdrantClient):
    """
    Create the Qdrant collection if it doesn't already exist.
    """
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        # Create payload index on list_name for faster filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="list_name",
            field_schema="keyword",
        )
        print(f"Created collection and index on list_name: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def upsert_places(client: QdrantClient, places: list[dict]):
    """
    Upload all embedded places into Qdrant.
    Each place becomes a point with its embedding vector
    and full metadata stored as payload (including list_name).
    """
    points = []
    for i, place in enumerate(places):
        payload = {k: v for k, v in place.items() if k != "embedding"}
        points.append(
            PointStruct(
                id=i,
                vector=place["embedding"],
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} places into Qdrant.")


def search(
    client: QdrantClient,
    query_vector: list[float],
    top_k: int = 5,
    list_name: str | None = None,
) -> list[dict]:
    """
    Semantic search with optional list_name filter.
    list_name: "visited", "want_to_go", or None (search both lists)
    """
    query_filter = None
    if list_name:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="list_name",
                    match=MatchValue(value=list_name),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload for hit in results.points]
