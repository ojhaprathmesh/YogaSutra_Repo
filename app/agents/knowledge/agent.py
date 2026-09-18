"""
app/agents/knowledge/agent.py — Knowledge / RAG Agent

ReAct-style agent that calls retrieve_yoga_knowledge() to fetch relevant
evidence chunks, then selects and summarises the most useful ones.

The agent writes retrieved evidence metadata to session state under
"retrieved_evidence" via output_key.
"""
from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from app.tools.retrieval import retrieve_yoga_knowledge

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-2.0-flash")

KNOWLEDGE_INSTRUCTION = """You are the Knowledge Retrieval Agent for YogaSutra, an AI yoga assistant.

You have access to a yoga knowledge base via the `retrieve_yoga_knowledge` tool.

Your job:
1. Read the user's request from the conversation.
2. Call `retrieve_yoga_knowledge` with a relevant query (e.g. "beginner relaxation morning 20 minutes").
3. Review the returned chunks.
4. Select the 3–5 most relevant chunks.
5. Return ONLY a JSON array of evidence chunk objects in this format:

[
  {
    "chunk_id": "doc_001_001",
    "document_id": "doc_001",
    "title": "Beginner's Guide to Morning Yoga",
    "section": "Introduction",
    "topic": "beginner_practice",
    "content": "...(full content)...",
    "source_type": "reference",
    "license": "curated",
    "relevance_score": 0.95
  }
]

Return ONLY the JSON array. No explanations, no markdown fences, no extra text.
"""

knowledge_agent = LlmAgent(
    name="KnowledgeAgent",
    model=_MODEL,
    description=(
        "Retrieves relevant yoga knowledge chunks from the knowledge base using "
        "the retrieve_yoga_knowledge tool. Returns evidence grounded in source metadata."
    ),
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[retrieve_yoga_knowledge],
    output_key="retrieved_evidence",
)
