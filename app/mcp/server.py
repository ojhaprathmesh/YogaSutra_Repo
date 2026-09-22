"""
app/mcp/server.py — Standard YogaSutra Model Context Protocol (MCP) Server

Exposes structured yoga domain capabilities via the standardized MCP tool interface:
1. 'search_yoga_knowledge': Semantic and keyword retrieval over curated classical corpus.
2. 'get_asana_metadata': Sanskrit name, alignment cues, modifications, and contraindications.
3. 'get_source_metadata': Citations, provenance, chapter/verse tracking.
4. 'get_practice_constraints': Medical/biomechanical cautions for injuries (back, knee, neck, hypertension).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.tools.retrieval import _CORPUS, retrieve_yoga_knowledge
from app.tools.safety import CONSTRAINT_KEYWORDS, _load_pose_database

logger = logging.getLogger("yogasutra.mcp.server")

_POSE_MAP = _load_pose_database()


class YogaSutraMCPServer:
    """Standardized MCP Server exposing YogaSutra tools."""

    def __init__(self) -> None:
        self.tools = {
            "search_yoga_knowledge": self.search_yoga_knowledge,
            "get_asana_metadata": self.get_asana_metadata,
            "get_source_metadata": self.get_source_metadata,
            "get_practice_constraints": self.get_practice_constraints,
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Returns JSON schema definitions of all exposed MCP tools."""
        return [
            {
                "name": "search_yoga_knowledge",
                "description": "Retrieve grounded classical yoga evidence chunks for a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_asana_metadata",
                "description": "Get detailed alignment, Sanskrit name, and contraindications for a pose.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asana_name": {"type": "string", "description": "English or Sanskrit name of the asana"},
                    },
                    "required": ["asana_name"],
                },
            },
            {
                "name": "get_source_metadata",
                "description": "Get provenance, source document, and line mapping for a knowledge chunk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk identifier, e.g. doc_001_001"},
                    },
                    "required": ["chunk_id"],
                },
            },
            {
                "name": "get_practice_constraints",
                "description": "Get contraindicated poses and safety precautions for a specific health condition.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "injury_type": {"type": "string", "description": "e.g. knee, neck, back, hypertension"},
                    },
                    "required": ["injury_type"],
                },
            },
        ]

    def search_yoga_knowledge(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return retrieve_yoga_knowledge(query, top_k=top_k)

    def get_asana_metadata(self, asana_name: str) -> dict[str, Any]:
        target = asana_name.strip().lower()
        if target in _POSE_MAP:
            return _POSE_MAP[target]
        for name, data in _POSE_MAP.items():
            if target in name or name in target:
                return data
        return {"error": f"Asana '{asana_name}' not found in knowledge corpus"}

    def get_source_metadata(self, chunk_id: str) -> dict[str, Any]:
        for chunk in _CORPUS:
            if chunk.get("chunk_id") == chunk_id:
                return {
                    "chunk_id": chunk_id,
                    "title": chunk.get("title"),
                    "source_file": chunk.get("source_file", "data/knowledge/corpus.json"),
                    "line_range": chunk.get("line_range", "1-50"),
                    "topic": chunk.get("topic"),
                }
        return {"error": f"Chunk ID '{chunk_id}' not found"}

    def get_practice_constraints(self, injury_type: str) -> dict[str, Any]:
        key = injury_type.strip().lower()
        contraindicated_poses: list[str] = []

        for name, pose in _POSE_MAP.items():
            cautions = " ".join(pose.get("contraindications", [])).lower()
            if key in cautions:
                contraindicated_poses.append(pose.get("title", name))

        return {
            "injury_type": injury_type,
            "caution_keywords": CONSTRAINT_KEYWORDS.get(key, [key]),
            "contraindicated_poses": list(set(contraindicated_poses)),
            "safe_modifications": f"Use props, avoid deep compressive stress on {injury_type}.",
        }

    def handle_rpc_request(self, rpc_request: dict[str, Any]) -> dict[str, Any]:
        """Handles standard JSON-RPC 2.0 requests."""
        req_id = rpc_request.get("id")
        method = rpc_request.get("method")
        params = rpc_request.get("params", {})

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}

        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            if tool_name not in self.tools:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool '{tool_name}' not found"},
                }
            try:
                result = self.tools[tool_name](**tool_args)
                return {"jsonrpc": "2.0", "id": req_id, "result": {"content": result}}
            except Exception as e:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32000, "message": str(e)},
                }

        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}


# Global singleton MCP server
mcp_server = YogaSutraMCPServer()
