# Maps RAG — Search Your Saved Google Maps Restaurants

A RAG (Retrieval Augmented Generation) application that lets you search your personal Google Maps saved places list using natural language.

**Ask things like:**
- "Any good Japanese restaurants that aren't too expensive?"
- "Italian places I've noted as good for dates"
- "What Korean BBQ spots have I saved?"

## Tech Stack
- **Python / Flask** — backend and web interface
- **Google Gemini** (`gemini-embedding-001`) — vector embeddings
- **Qdrant Cloud** — vector database
- **Google Places API** — restaurant enrichment (hours, price, cuisine)

## Setup

### 1. Clone and install dependencies
```bash
git clone <your-repo>
cd maps-rag
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Fill in your keys in .env
```

**Keys needed:**
- `GEMINI_API_KEY` — from [Google AI Studio](https://aistudio.google.com)
- `GOOGLE_PLACES_API_KEY` — from [Google Cloud Console](https://console.cloud.google.com)
- `QDRANT_URL` and `QDRANT_API_KEY` — from [Qdrant Cloud](https://qdrant.io) free tier

### 3. Export your Google Maps data
1. Go to [takeout.google.com](https://takeout.google.com)
2. Deselect all, then select only **Maps**
3. Download and extract
4. Copy `Saved Places.json` to `data/raw/Saved Places.json`

### 4. Build the index (run once)
```bash
python scripts/populate_qdrant.py
```

### 5. Run the app
```bash
python src/app.py
```
Visit `http://localhost:5000`

## Project Structure
```
maps-rag/
├── data/raw/              # Your Takeout export (gitignored)
├── src/
│   ├── ingest.py          # Parse Takeout + enrich with Places API
│   ├── embed.py           # Gemini embeddings
│   ├── index.py           # Qdrant storage and retrieval
│   ├── query.py           # RAG pipeline
│   └── app.py             # Flask web interface
└── scripts/
    └── build_index.py     # One-time indexing script
```
