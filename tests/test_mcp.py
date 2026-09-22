"""
tests/test_mcp.py — Unit tests for YogaSutra MCP Server and Client
"""
import pytest
from app.mcp.server import YogaSutraMCPServer
from app.mcp.client import YogaSutraMCPClient


class TestMCPServerAndClient:
    def test_mcp_list_tools(self):
        server = YogaSutraMCPServer()
        client = YogaSutraMCPClient(server=server)
        tools = client.list_tools()
        names = [t["name"] for t in tools]
        assert "search_yoga_knowledge" in names
        assert "get_asana_metadata" in names
        assert "get_source_metadata" in names
        assert "get_practice_constraints" in names

    def test_mcp_search_yoga_knowledge(self):
        server = YogaSutraMCPServer()
        client = YogaSutraMCPClient(server=server)
        results = client.call_tool("search_yoga_knowledge", {"query": "morning beginner", "top_k": 3})
        assert isinstance(results, list)
        assert len(results) > 0
        assert "chunk_id" in results[0]

    def test_mcp_get_asana_metadata(self):
        server = YogaSutraMCPServer()
        client = YogaSutraMCPClient(server=server)
        data = client.call_tool("get_asana_metadata", {"asana_name": "Balasana"})
        assert "title" in data or "sanskrit_name" in data

    def test_mcp_get_practice_constraints(self):
        server = YogaSutraMCPServer()
        client = YogaSutraMCPClient(server=server)
        res = client.call_tool("get_practice_constraints", {"injury_type": "knee"})
        assert res["injury_type"] == "knee"
        assert "contraindicated_poses" in res
