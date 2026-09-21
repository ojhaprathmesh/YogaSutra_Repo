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

_MODEL = os.environ.get("YOGASUTRA_MODEL", "gemini-3.5-flash-lite")

KNOWLEDGE_INSTRUCTION = """You are the Knowledge Retrieval Agent for YogaSutra, an AI yoga assistant.

You have access to a yoga knowledge base via the `retrieve_yoga_knowledge` tool.

Your job:
1. Carefully analyze the user's request and 'user_profile' in session state. Identify ALL specific goals, physical conditions, postural concerns, or limitations (e.g. "flat feet", "rounded shoulders", "morning relaxation", "lower back pain").
2. Call `retrieve_yoga_knowledge` with targeted queries covering EACH condition. If the user mentions multiple specific areas (e.g. flat feet AND rounded shoulders), search for BOTH (e.g. "flat feet arch strengthening pada bandha", "rounded shoulders chest opening posture").
3. Review the returned chunks and select the 4–6 most relevant and diverse chunks so that ALL the user's stated concerns are backed by authentic evidence.
4. Return ONLY a JSON array of evidence chunk objects in this format:

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
