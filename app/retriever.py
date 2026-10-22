"""
Retrieval layer — semantic search over ingested document chunks.

Responsibility: given a natural-language query, return the ``top_k`` most
semantically similar chunks from ChromaDB, formatted for downstream use.

Design decisions:
  - Distance is rounded to 4 d.p. — enough precision for ranking without
    noise from floating-point representation.
  - ``format_context`` separates chunks with a horizontal rule so the LLM
    can clearly distinguish where one source ends and another begins.
  - The collection client is instantiated per-call (stateless function).
    In a high-throughput service you would inject a shared client; for this
    use-case the overhead is negligible and keeps the module easy to test.
"""

import logging
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions

from app.config import settings
from app.exceptions import ConfigurationError, RetrievalError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_api_key() -> None:
    if not settings.openai_api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. Required for embedding the query."
        )


def get_collection() -> chromadb.Collection:
    """Return the ChromaDB collection used for retrieval."""
    _require_api_key()
    client = chromadb.PersistentClient(path=settings.chroma_path)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    return client.get_or_create_collection(
        name=settings.collection_name,
        embedding_function=ef,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = settings.default_top_k) -> List[Dict]:
    """
    Return the ``top_k`` most relevant document chunks for a query.

    Each result dict contains:
    - ``text``        — the raw chunk text
    - ``source``      — filename the chunk came from
    - ``chunk_index`` — position of the chunk within that file
    - ``distance``    — cosine distance (lower = more similar)

    Args:
        query:  Natural-language question or search string.
        top_k:  Number of chunks to return. Capped at ``settings.max_top_k``.

    Returns:
        List of chunk dicts, ordered by ascending distance.

    Raises:
        RetrievalError: If the ChromaDB query fails.
    """
    top_k = min(top_k, settings.max_top_k)

    try:
        collection = get_collection()
        logger.info(collection)
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        logger.info(results)
    except Exception as exc:
        raise RetrievalError(f"ChromaDB query failed: {exc}") from exc

    chunks: List[Dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk", 0),
                "distance": round(dist, 4),
            }
        )
    logger.info(chunks)

    logger.debug("Retrieved %d chunks for query: %.80s", len(chunks), query)
    return chunks


def format_context(chunks: List[Dict]) -> str:
    """
    Render retrieved chunks as a single, clearly labelled context block.

    Each chunk is prefixed with its source filename and position so the LLM
    can cite sources accurately.  Chunks are separated by a horizontal rule
    to minimise context bleed between sources.

    Args:
        chunks: List of chunk dicts as returned by :func:`retrieve`.

    Returns:
        Formatted string ready to embed in an LLM prompt.
    """
    sections = [
        f"[Source {i}: {chunk['source']} — chunk {chunk['chunk_index']}]\n{chunk['text']}"
        for i, chunk in enumerate(chunks, 1)
    ]
    return "\n\n---\n\n".join(sections)
