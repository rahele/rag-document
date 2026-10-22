# RAG Document Q&A

A production-style Retrieval-Augmented Generation (RAG) API that allows users to ask questions against their own documents. Built with **FastAPI**, **Anthropic Claude**, **OpenAI embeddings**, and **ChromaDB**.

## Architecture

```
User question
     │
     ▼
FastAPI /ask endpoint
     │
     ▼
ChromaDB (semantic search over embedded document chunks)
     │  returns top-k relevant chunks
     ▼
Anthropic Claude API (answer grounded in retrieved context)
     │
     ▼
Response: answer + sources + token usage
```

## Features

- **Document ingestion**: Load `.txt` files, split into chunks, embed and store in ChromaDB
- **Semantic retrieval**: Query ChromaDB for the most relevant chunks given a user question
- **Grounded answers**: Claude answers strictly from retrieved context — no hallucination from general knowledge
- **Source attribution**: Every answer includes which documents and chunks were used
- **Token tracking**: Input/output token counts returned per request
- **Input validation**: Pydantic models enforce request structure
- **Test coverage**: Unit and integration tests with mocked external dependencies

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

## Design Decisions

- **Chunking strategy**: `RecursiveCharacterTextSplitter` with 800-token chunks and 100-token overlap preserves sentence and paragraph boundaries
- **Embeddings**: OpenAI `text-embedding-3-small` — fast and cost-effective for document search
- **LLM**: Anthropic Claude with a strict system prompt that prevents answers outside the provided context
- **Upsert**: Documents are upserted (not inserted) so re-ingestion is safe and idempotent
- **Source IDs**: Chunk IDs are deterministic (MD5 of filepath + chunk index) enabling reliable upsert
