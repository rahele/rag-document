import hashlib
import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import settings
from app.exceptions import ConfigurationError, IngestionError

logger = logging.getLogger(__name__)


def _require_api_key() -> None:
    """Raise ConfigurationError if the OpenAI key is missing."""
    if not settings.openai_api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )


def get_collection() -> chromadb.Collection:
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


def _chunk_id(filepath: str, chunk_index: int) -> str:
    h = hashlib.md5(filepath.encode()).hexdigest()[:8]
    return f"{h}-{chunk_index}"


def _build_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )


def ingest_file(filepath: Path, collection: chromadb.Collection) -> int:
    """
    Ingest a single text file into ChromaDB.

    Args:
        filepath: Absolute or relative path to a UTF-8 text file.
        collection: Target ChromaDB collection.

    Returns:
        Number of chunks upserted.

    Raises:
        IngestionError: If the file cannot be read or the upsert fails.
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestionError(f"Cannot read {filepath}: {exc}") from exc

    splitter = _build_splitter()
    chunks = splitter.split_text(text)


    if not chunks:
        logger.warning("No chunks produced for %s — file may be empty.", filepath.name)
        return 0

    ids = [_chunk_id(str(filepath), i) for i in range(len(chunks))]
    metadatas = [{"source": filepath.name, "chunk": i} for i in range(len(chunks))]

    try:
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    except Exception as exc:
        raise IngestionError(f"ChromaDB upsert failed for {filepath.name}: {exc}") from exc

    logger.info("Ingested %s: %d chunks stored.", filepath.name, len(chunks))
    return len(chunks)


def ingest_all() -> dict:
    docs_dir = Path(settings.documents_path)
    if not docs_dir.exists():
        raise IngestionError(
            f"Documents directory not found: {docs_dir.resolve()}. "
            "Create it and add .txt files before ingesting."
        )

    collection = get_collection()
    txt_files = list(docs_dir.glob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in %s.", docs_dir.resolve())
        return {"files": 0, "chunks": 0}

    total_chunks = 0
    for filepath in sorted(txt_files):
        total_chunks += ingest_file(filepath, collection)

    logger.info(
        "Ingestion complete: %d file(s), %d total chunks.", len(txt_files), total_chunks
    )
    return {"files": len(txt_files), "chunks": total_chunks}


if __name__ == "__main__":
    # Allow running directly: `python -m app.ingest`
    from app.logging_config import configure_logging
    configure_logging(settings.log_level)
    summary = ingest_all()
    print(f"Done — {summary['files']} files, {summary['chunks']} chunks.")
