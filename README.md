
## Setup

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Add your documents to the documents/ folder (plain .txt files)

# Ingest documents into ChromaDB
python -m app.ingest

# Start the API
uvicorn app.main:app --reload
```

## API Endpoints

### `GET /health`
Health check.

### `POST /ingest`
Ingest all `.txt` documents from the `documents/` directory.

### `POST /ask`
Ask a question against the ingested documents.

**Request:**
```json
{
  "question": "What is the cancellation policy?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "What is the cancellation policy?",
  "answer": "According to sample_policy.txt, clients may cancel within 14 calendar days...",
  "sources": [
    {
      "source": "sample_policy.txt",
      "chunk_index": 0,
      "distance": 0.1234,
      "preview": "Clients may cancel their service agreement within 14 calendar days..."
    }
  ],
  "model": "claude-3-5-sonnet-20241022",
  "input_tokens": 320,
  "output_tokens": 45
}
```

## Running Tests

```bash
pytest tests/ -v
```
