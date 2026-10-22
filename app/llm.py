"""
LLM integration layer — sends retrieved context + user question to Claude.

Responsibility: construct a well-constrained prompt, call the Anthropic
Messages API, and return the answer with full token usage metadata.

Design decisions:
  - Strict system prompt with explicit rules prevents the model from
    hallucinating or drifting outside the document context.
  - Token usage is returned in every response so callers can track cost
    and enforce budget limits at the application layer.
  - The client is instantiated per-call (stateless). For high throughput,
    inject a shared ``anthropic.Anthropic`` instance via dependency injection.
  - ``ANTHROPIC_API_KEY`` is validated at call time rather than import time
    so the app can start without the key (useful for health checks / CI).
"""

import logging
from typing import Dict

import anthropic

from app.config import settings
from app.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a precise, source-grounded assistant. Your sole function is to answer \
questions using only the document excerpts supplied in the context block below.

Rules — follow all of them without exception:
1. Base every statement on the provided context. Do not draw on outside knowledge.
2. If the context does not contain sufficient information to answer the question, say so explicitly \
   (e.g. "The provided documents do not contain information about X.").
3. Cite the source document by name where relevant \
   (e.g. "According to policy_v3.txt, …").
4. Do not speculate, infer beyond what is written, or fill gaps with assumptions.
5. Keep the answer concise, accurate, and free of filler phrases."""


def _require_api_key() -> None:
    if not settings.anthropic_api_key:
        raise ConfigurationError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )


def ask(question: str, context: str) -> Dict:
    """
    Send a question with retrieved document context to Claude and return the answer.

    The user message is structured to clearly separate the context block from the
    question, reducing the chance of the model conflating the two.

    Args:
        question: The user's natural-language question.
        context:  Formatted document chunks as returned by
                  :func:`app.retriever.format_context`.

    Returns:
        Dict with keys:
        - ``answer``        — Claude's response text
        - ``model``         — exact model identifier used (as reported by the API)
        - ``input_tokens``  — tokens consumed by the prompt
        - ``output_tokens`` — tokens consumed by the completion

    Raises:
        ConfigurationError: If ``ANTHROPIC_API_KEY`` is missing.
        LLMError:           If the Anthropic API call fails.
    """
    _require_api_key()

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    user_message = (
        "Context from documents:\n\n"
        f"{context}\n\n"
        "---\n\n"
        f"Question: {question}"
    )

    logger.debug("Sending request to %s (max_tokens=%d).", settings.claude_model, settings.llm_max_tokens)

    try:
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.llm_max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"Could not connect to Anthropic API: {exc}") from exc

    answer = response.content[0].text
    logger.info(
        "LLM response received. model=%s input_tokens=%d output_tokens=%d",
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    return {
        "answer": answer,
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
