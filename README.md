# SHL Assessment Recommender

A conversational FastAPI agent that recommends SHL talent assessments through dialogue.

## Architecture

```
SHL Catalog (scraped, 389 Individual Test Solutions)
  → ChromaDB (semantic) + BM25 (keyword) → RRF fusion
  → LangGraph-style agent (Gemini 2.0 Flash)
  → FastAPI (POST /chat, GET /health)
  → Deployed on Render (Docker, free tier)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY=your_key_here   # Linux/Mac
$env:GEMINI_API_KEY="your_key_here"  # Windows PowerShell

# Run the server
uvicorn app.main:app --reload --port 8000
```

## API

### GET /health
```json
{"status": "ok"}
```

### POST /chat
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a mid-level Java developer"}
  ]
}
```
Response:
```json
{
  "reply": "...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"}
  ],
  "end_of_conversation": false
}
```

## Tests

```bash
pytest tests/ -v
```

## Scraping (if you need to refresh the catalog)

```bash
pip install -r scraper/requirements.txt
python scraper/scraper.py
```
