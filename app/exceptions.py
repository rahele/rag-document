
class RAGException(Exception):
    """Base class for all RAG pipeline exceptions."""


class IngestionError(RAGException):
    """Raised when a document cannot be ingested (read, split, or embedded)."""


class RetrievalError(RAGException):
    """Raised when ChromaDB cannot be queried."""


class LLMError(RAGException):
    """Raised when the Anthropic API call fails."""


class ConfigurationError(RAGException):
    """Raised when a required configuration value is missing or invalid."""
