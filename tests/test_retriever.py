from unittest.mock import MagicMock, patch
from app.retriever import retrieve, format_context


MOCK_QUERY_RESULT = {
    "documents": [["First relevant chunk text.", "Second relevant chunk text."]],
    "metadatas": [
        [
            {"source": "policy.txt", "chunk": 0},
            {"source": "contract.txt", "chunk": 2},
        ]
    ],
    "distances": [[0.1523, 0.2841]],
}


@patch("app.retriever.get_collection")
def test_retrieve_returns_chunks(mock_get_collection):
    mock_collection = MagicMock()
    mock_collection.query.return_value = MOCK_QUERY_RESULT
    mock_get_collection.return_value = mock_collection

    results = retrieve("What is the cancellation policy?", top_k=2)

    assert len(results) == 2
    assert results[0]["source"] == "policy.txt"
    assert results[0]["distance"] == 0.1523
    assert "First relevant chunk" in results[0]["text"]


@patch("app.retriever.get_collection")
def test_retrieve_calls_query_with_correct_params(mock_get_collection):
    mock_collection = MagicMock()
    mock_collection.query.return_value = MOCK_QUERY_RESULT
    mock_get_collection.return_value = mock_collection

    retrieve("test question", top_k=3)

    mock_collection.query.assert_called_once_with(
        query_texts=["test question"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )


def test_format_context_empty_list():
    context = format_context([])
    assert context == ""
