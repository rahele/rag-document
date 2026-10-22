"""
RAG Document Q&A — FastAPI application entry point.

Endpoints:
  GET  /health  — liveness check; returns 200 when the service is up
  POST /ingest  — ingest all documents from the configured documents/ folder
  POST /ask     — answer a question grounded in the ingested documents

"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.exceptions import ConfigurationError, IngestionError, LLMError, RetrievalError
from app.ingest import ingest_all
from app.llm import ask
from app.logging_config import configure_logging
from app.retriever import format_context, retrieve

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


# Lifespan — startup / shutdown hooks

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run setup tasks on startup and cleanup on shutdown."""
    logger.info(
        "RAG Document Q&A starting. model=%s chroma_path=%s",
        settings.claude_model,
        settings.chroma_path,
    )
    yield
    logger.info("RAG Document Q&A shut down.")


# Application

app = FastAPI(
    title="RAG Document Q&A",
    description=(
        "Ask questions against your own documents "
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Middleware — attach a correlation ID to every request

@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    """
    Attach a unique ``X-Request-ID`` header to every response.

    Clients can echo this ID in support requests so that server-side log
    entries for the same request can be found immediately.
    """
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Exception handlers — map domain exceptions to HTTP responses

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    logger.error("Configuration error: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc), "type": "configuration_error"})


@app.exception_handler(IngestionError)
async def ingestion_error_handler(request: Request, exc: IngestionError):
    logger.error("Ingestion error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": "ingestion_error"})


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request: Request, exc: RetrievalError):
    logger.error("Retrieval error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc), "type": "retrieval_error"})


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    logger.error("LLM error: %s", exc)
    return JSONResponse(status_code=502, content={"detail": str(exc), "type": "llm_error"})


# Request / Response schemas

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Natural-language question to answer.")
    top_k: Optional[int] = Field(
        settings.default_top_k,
        ge=1,
        le=settings.max_top_k,
        description="Number of document chunks to retrieve as context.",
    )

    model_config = {"json_schema_extra": {"example": {"question": "What is the refund policy?", "top_k": 5}}}


class SourceChunk(BaseModel):
    source: str = Field(..., description="Filename of the source document.")
    chunk_index: int = Field(..., description="Zero-based position of this chunk within the source file.")
    distance: float = Field(..., description="Cosine distance from the query (lower = more relevant).")
    preview: str = Field(..., description="First 200 characters of the chunk text.")


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    model: str = Field(..., description="Exact model ID used for generation.")
    input_tokens: int
    output_tokens: int


class IngestResponse(BaseModel):
    status: str
    files_ingested: int
    chunks_stored: int
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str


# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health():
    """
    Liveness probe.

    Returns 200 when the service process is running. 
    """
    return HealthResponse(status="ok", version=app.version)


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
def ingest():
    """
    Ingest all ``.txt`` documents from the configured documents directory.

    """
    logger.info("Ingestion requested.")
    summary = ingest_all()
    return IngestResponse(
        status="success",
        files_ingested=summary["files"],
        chunks_stored=summary["chunks"],
        message=f"Ingested {summary['files']} file(s), {summary['chunks']} chunk(s) stored.",
    )


@app.post("/ask", response_model=QuestionResponse, tags=["Q&A"])
def ask_question(request: QuestionRequest):
    """
    Answer a question using retrieved document context and Claude.

    Workflow:
    1. Embed the question and retrieve the ``top_k`` most relevant chunks.
    2. Format the chunks into a context block.
    3. Send the context + question to Claude with a strict grounding prompt.
    4. Return the answer, source attribution, and token usage.

    Returns 404 if no relevant documents are found in the vector store.
    """
    logger.info("Question received: %.120s", request.question)

    chunks = retrieve(request.question, top_k=request.top_k)
    if not chunks:
        logger.warning("No chunks retrieved for question: %.120s", request.question)
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Ensure documents have been ingested via POST /ingest.",
        )

    context = format_context(chunks)
    result = ask(request.question, context)

    sources = [
        SourceChunk(
            source=c["source"],
            chunk_index=c["chunk_index"],
            distance=c["distance"],
            preview=c["text"][:200] + "…" if len(c["text"]) > 200 else c["text"],
        )
        for c in chunks
    ]

    logger.info(
        "Answer delivered. sources=%d input_tokens=%d output_tokens=%d",
        len(sources),
        result["input_tokens"],
        result["output_tokens"],
    )

    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        sources=sources,
        model=result["model"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )
