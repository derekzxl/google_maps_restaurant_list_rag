"""
query.py
The core RAG pipeline. Takes a natural language query,
detects which list to filter on, retrieves relevant places
from Qdrant, and uses Gemini to synthesize a helpful answer.
"""

import os
from google import genai
from dotenv import load_dotenv
from src.embed import embed_query
from src.index import get_client, search

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
GENERATIVE_MODEL = "gemini-2.0-flash"


import json
from pydantic import BaseModel
from typing import Literal
from google.genai import types

class QueryClassifier(BaseModel):
    list_name: Literal["visited", "want_to_go", "null"]


def detect_list_filter(user_query: str) -> str | None:
    """
    Ask Gemini to determine which list filter to apply based on the query.
    Returns "visited", "want_to_go", or None (search both).
    """
    prompt = f"""
You are a classifier. Given a user's restaurant search query, determine which list they are searching:
- "visited": they want restaurants they have already been to (e.g. "remind me", "places I've been", "where I ate", "I went to")
- "want_to_go": they want restaurants they haven't been to yet (e.g. "suggest", "recommend", "haven't tried", "want to go", "new places")
- null: the query is ambiguous or could apply to both lists

Query: {user_query}
""".strip()

    try:
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryClassifier,
            ),
        )
        data = json.loads(response.text)
        result = data.get("list_name")
    except Exception as e:
        print(f"  Warning: Classifier failed, falling back to searching both lists. Error: {e}")
        result = "null"

    if result == "visited":
        return "visited"
    elif result == "want_to_go":
        return "want_to_go"
    return None


def format_places_for_context(places: list[dict]) -> str:
    """
    Format retrieved places into a readable context block
    to inject into the LLM prompt.
    """
    price_map = {0: "free", 1: "inexpensive", 2: "moderate", 3: "expensive", 4: "very expensive"}

    sections = []
    for p in places:
        price_str = price_map.get(p.get("price_level"), "unknown")
        hours = "\n    ".join(p.get("hours", [])) or "unknown"
        list_label = "✓ Been here" if p.get("list_name") == "visited" else "★ Want to go"

        business_status = p.get("business_status", "OPERATIONAL")
        if business_status == "CLOSED_PERMANENTLY":
            status_str = "⛔ Permanently closed"
        elif business_status == "CLOSED_TEMPORARILY":
            status_str = "⚠️ Temporarily closed"
        else:
            status_str = "✅ Open for business"

        section = f"""
{list_label}
Name: {p.get('name')}
Status: {status_str}
Address: {p.get('address')}
Cuisine / types: {', '.join(p.get('types', []))}
Price: {price_str}
Rating: {p.get('rating', 'unknown')} / 5
Open now: {p.get('open_now', 'unknown')}
Hours:
    {hours}
Your note: {p.get('note') or 'none'}
Your tags: {', '.join(p.get('tags', [])) or 'none'}
        """.strip()
        sections.append(section)

    return "\n\n---\n\n".join(sections)


def answer_query(user_query: str, top_k: int = 5) -> str:
    """
    Full RAG pipeline:
    1. Detect which list to filter on
    2. Embed the user query
    3. Retrieve top_k similar places from Qdrant (with filter)
    4. Format retrieved places as context
    5. Ask Gemini to answer using only that context
    Returns a natural language answer string.
    """
    try:
        # Step 1: detect list filter
        list_filter = detect_list_filter(user_query)
        filter_label = list_filter or "both lists"
        print(f"  List filter detected: {filter_label}")

        # Step 2: embed the query
        query_vector = embed_query(user_query)

        # Step 3: retrieve
        qdrant_client = get_client()
        retrieved = search(qdrant_client, query_vector, top_k=top_k, list_name=list_filter)
    except Exception as e:
        print(f"Error during retrieval: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "Unable to perform search because the Gemini API quota/credits are depleted. Please check your Gemini billing details."
        return "Failed to query the restaurant database. Please check your Qdrant or API connection."

    if not retrieved:
        return "I couldn't find any matching restaurants in your saved lists."

    # Step 4: format context
    context = format_places_for_context(retrieved)

    # Step 5: ask Gemini
    prompt = f"""
You are a helpful assistant with access to a user's personal saved restaurant lists.
They have two lists: "visited" (places they've already been to) and "want_to_go" (places they want to try).

Answer the user's question using ONLY the restaurant information provided below.
Be conversational, specific, and helpful. Reference the user's own notes when relevant.
If the information needed isn't available in the data, say so honestly.

RESTAURANTS:
{context}

USER QUESTION: {user_query}

ANSWER:
    """.strip()

    try:
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=prompt,
        )
        return response.text
    except Exception as e:
        print(f"Error during response generation: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return "Unable to generate answer because the Gemini API quota/credits are depleted. Please check your Gemini billing details."
        return "Failed to generate an answer from Gemini. Please try again later."


if __name__ == "__main__":
    print(answer_query("Suggest a cheap Japanese restaurant I haven't tried yet"))
    print("---")
    print(answer_query("Remind me any places I've been to where portions were too small"))
