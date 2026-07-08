"""
embed.py
Generates vector embeddings for place documents using
Google's Gemini gemini-embedding-001 model.
"""

import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"


from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# We retry on any exception just in case, but you can also specifically catch google.genai.errors.ClientError
@retry(
    wait=wait_exponential(multiplier=1, min=30, max=120),
    stop=stop_after_attempt(10),
    reraise=True
)
def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.
    Returns a list of floats representing the vector.
    Automatically retries with exponential backoff if a rate limit is hit.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values

# We retry on any exception just in case, but you can also specifically catch google.genai.errors.ClientError
@retry(
    wait=wait_exponential(multiplier=1, min=30, max=120),
    stop=stop_after_attempt(10),
    reraise=True
)
def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user search query.
    Uses RETRIEVAL_QUERY task type for better search performance.
    """
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def embed_places(places: list[dict]) -> list[dict]:
    """
    Add an embedding vector to each place dict.
    Embeds the document_text field generated during ingestion.
    Returns the same list with 'embedding' field added to each place.
    """
    for i, place in enumerate(places):
        text = place.get("document_text", "")
        time.sleep(10)
        place["embedding"] = embed_text(text)
        print(f"  [{i+1}/{len(places)}] Embedded: {place['name']}") 
    return places
