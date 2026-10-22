import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.exceptions import IngestionError

client = TestClient(app, raise_server_exceptions=False)


# Test fixtures

MOCK_CHUNKS = [
    {
        "text": "The cancellation policy allows clients to cancel within 14 days of purchase.",
        "source": "policy.txt",
        "chunk_index": 0,
        "distance": 0.12,
    }
]

MOCK_LLM_RESULT = {
    "answer": "According to policy.txt, clients may cancel within 14 days.",
    "model": "claude-3-5-sonnet-20241022",
    "input_tokens": 320,
    "output_tokens": 45,
}


# /health

class TestHealth:
    def test_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_response_contains_status_and_version(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data


# /ingest

class TestIngest:
    @patch("app.main.ingest_all", return_value={"files": 3, "chunks": 42})
    def test_successful_ingestion(self, mock_ingest):
        response = client.post("/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["files_ingested"] == 3
        assert data["chunks_stored"] == 42

    @patch("app.main.ingest_all", side_effect=IngestionError("Documents directory not found"))
    def test_ingestion_error_returns_500(self, mock_ingest):
        response = client.post("/ingest")
        assert response.status_code == 500
        assert "ingestion_error" in response.json()["type"]


# /ask

class TestAsk:
    @patch("app.main.retrieve", return_value=MOCK_CHUNKS)
    @patch("app.main.ask", return_value=MOCK_LLM_RESULT)
    def test_successful_question(self, mock_ask, mock_retrieve):
        response = client.post("/ask", json={"question": "What is the cancellation policy?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == MOCK_LLM_RESULT["answer"]
        assert data["input_tokens"] == 320
        assert data["output_tokens"] == 45
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source"] == "policy.txt"

    @patch("app.main.retrieve", return_value=MOCK_CHUNKS)
    @patch("app.main.ask", return_value=MOCK_LLM_RESULT)
    def test_response_includes_source_attribution(self, mock_ask, mock_retrieve):
        response = client.post("/ask", json={"question": "What is the cancellation policy?"})
        source = response.json()["sources"][0]
        assert "source" in source
        assert "chunk_index" in source
        assert "distance" in source
        assert "preview" in source

    @patch("app.main.retrieve", return_value=[])
    def test_returns_404_when_no_chunks_found(self, mock_retrieve):
        response = client.post("/ask", json={"question": "Something with no matching documents"})
        assert response.status_code == 404

    def test_rejects_empty_question(self):
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 422

    def test_rejects_question_below_min_length(self):
        response = client.post("/ask", json={"question": "hi"})
        assert response.status_code == 422

    def test_rejects_top_k_of_zero(self):
        response = client.post("/ask", json={"question": "valid question here", "top_k": 0})
        assert response.status_code == 422

    def test_rejects_top_k_above_maximum(self):
        response = client.post("/ask", json={"question": "valid question here", "top_k": 999})
        assert response.status_code == 422

    @patch("app.main.retrieve", return_value=MOCK_CHUNKS)
    @patch("app.main.ask", return_value=MOCK_LLM_RESULT)
    def test_custom_top_k_is_respected(self, mock_ask, mock_retrieve):
        response = client.post("/ask", json={"question": "What is the cancellation policy?", "top_k": 3})
        assert response.status_code == 200
        mock_retrieve.assert_called_once_with("What is the cancellation policy?", top_k=3)
