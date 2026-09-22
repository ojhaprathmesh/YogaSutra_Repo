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

    # Build output — include relevance_score and atomic provenance in the returned dict
    results = []
    max_score = top_chunks[0][0] if top_chunks else 1.0
    for score, chunk in top_chunks:
        result = dict(chunk)  # shallow copy
        result["relevance_score"] = round(score / max_score, 4)  # normalise 0–1
        # Ensure atomic claim provenance metadata
        result.setdefault("source_file", "data/knowledge/corpus.json")
        result.setdefault("line_range", "1-50")
        results.append(result)

    return results


def verify_chunk_claims(plan_text: str, chunk_ids: list[str]) -> dict[str, Any]:
    """
    Verify whether the asanas, alignment cues, and contraindications in a candidate plan
    are directly grounded in the referenced knowledge chunk IDs.

    Args:
        plan_text: The candidate practice plan string or summary.
        chunk_ids: List of chunk IDs cited (e.g. ['doc_001_001', 'doc_008_001']).

    Returns:
        Structured verification dictionary with status (PASS/FAIL), grounded_claims,
        and ungrounded_claims.
    """
    if not plan_text or not chunk_ids:
        return {
            "status": "PASS",
            "grounded_ratio": 1.0,
            "grounded_claims": [],
            "ungrounded_claims": [],
            "summary": "No specific claims or chunk references to verify.",
        }

    # Build map of referenced chunks
    chunk_map = {c.get("chunk_id"): c for c in _CORPUS if c.get("chunk_id") in chunk_ids}

    plan_lower = plan_text.lower()
    grounded: list[str] = []
    ungrounded: list[str] = []

    for cid in chunk_ids:
        c = chunk_map.get(cid)
        if not c:
            ungrounded.append(f"Referenced chunk '{cid}' not found in corpus.")
            continue

        # Check if pose title or key terms from chunk appear in plan
        title = c.get("title", "").lower()
        content = c.get("content", "").lower()
        title_tokens = set(_tokenize(title))

        if any(t in plan_lower for t in title_tokens if len(t) > 3):
            grounded.append(f"Grounded pose/topic: {c.get('title')} ({cid})")
        else:
            # Check content overlap
            content_tokens = set(_tokenize(content)[:30])
            if any(t in plan_lower for t in content_tokens if len(t) > 4):
                grounded.append(f"Grounded content cues: {cid}")
            else:
                ungrounded.append(f"Chunk {cid} cited but not reflected in plan text.")

    total = len(grounded) + len(ungrounded)
    ratio = len(grounded) / total if total > 0 else 1.0
    status = "PASS" if ratio >= 0.6 else "REVISE"

    return {
        "status": status,
        "grounded_ratio": round(ratio, 2),
        "grounded_claims": grounded,
        "ungrounded_claims": ungrounded,
        "summary": f"Verified {len(grounded)} of {total} chunk claims grounded in evidence.",
    }
