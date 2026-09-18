"""
app/tools/retrieval.py — Knowledge Retrieval Tool

Phase 1: in-memory RAG over the static corpus in data/knowledge/corpus.json.
No vector database required; uses simple keyword-based relevance scoring.

Tool contract:
    retrieve_yoga_knowledge(query: str, top_k: int = 5) -> list[dict]

Each returned dict matches the EvidenceChunk schema from app/state.py.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Corpus loading (loaded once at module import)
# ---------------------------------------------------------------------------

_CORPUS_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge" / "corpus.json"

def _load_corpus() -> list[dict[str, Any]]:
    with open(_CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


_CORPUS: list[dict[str, Any]] = _load_corpus()


# ---------------------------------------------------------------------------
# Simple TF-based relevance scorer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _score_chunk(chunk: dict[str, Any], query_tokens: set[str]) -> float:
    """
    Score a chunk against query tokens using term-frequency overlap.
    Also adds a small boost for topic/section keyword hits.
    """
    # Combine all searchable text fields
    searchable = " ".join([
        chunk.get("title", ""),
        chunk.get("section", ""),
        chunk.get("topic", ""),
        chunk.get("content", ""),
    ])
    chunk_tokens = _tokenize(searchable)
    if not chunk_tokens:
        return 0.0

    # TF-overlap score
    overlap = sum(1 for t in chunk_tokens if t in query_tokens)
    tf_score = overlap / math.sqrt(len(chunk_tokens))

    # Topic boost — if the topic keyword appears directly in the query
    topic = chunk.get("topic", "")
    topic_tokens = set(_tokenize(topic))
    topic_boost = 0.5 if topic_tokens & query_tokens else 0.0

    return tf_score + topic_boost


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def retrieve_yoga_knowledge(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant yoga knowledge chunks for the given query.

    Args:
        query:  Natural-language search query.
        top_k:  Maximum number of chunks to return (default 5).

    Returns:
        List of chunk dicts, each conforming to the EvidenceChunk schema,
        sorted by descending relevance_score. Returns at most top_k items.
    """
    if not query or not query.strip():
        return []

    query_tokens = set(_tokenize(query))

    # Score every chunk
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in _CORPUS:
        score = _score_chunk(chunk, query_tokens)
        if score > 0:
            scored.append((score, chunk))

    # Sort descending and take top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = scored[:top_k]

    # Build output — include relevance_score in the returned dict
    results = []
    max_score = top_chunks[0][0] if top_chunks else 1.0
    for score, chunk in top_chunks:
        result = dict(chunk)  # shallow copy
        result["relevance_score"] = round(score / max_score, 4)  # normalise 0–1
        results.append(result)

    return results
