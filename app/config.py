
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    claude_model: str = "claude-3-5-sonnet-20241022"
    embedding_model: str = "text-embedding-3-small"
    llm_max_tokens: int = 1024

    chroma_path: str = "./chroma_db"
    documents_path: str = "./documents"
    collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 100

    default_top_k: int = 5
    max_top_k: int = 20

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
