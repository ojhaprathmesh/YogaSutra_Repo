"""
app/mcp/client.py — YogaSutra MCP Client Adapter

Allows ADK agents or external services to invoke tools over the standard
Model Context Protocol JSON-RPC specification.
"""
from __future__ import annotations

import logging
from typing import Any

from app.mcp.server import YogaSutraMCPServer, mcp_server

logger = logging.getLogger("yogasutra.mcp.client")


class YogaSutraMCPClient:
    """Client communicating with YogaSutra MCP Server."""

    def __init__(self, server: YogaSutraMCPServer | None = None) -> None:
        self.server = server or mcp_server

    def list_tools(self) -> list[dict[str, Any]]:
        """Query available tools from the server."""
        rpc = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
        res = self.server.handle_rpc_request(rpc)
        return res.get("result", {}).get("tools", [])

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a named tool on the MCP server."""
        rpc = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }
        res = self.server.handle_rpc_request(rpc)
        if "error" in res:
            raise RuntimeError(f"MCP Tool '{tool_name}' error: {res['error'].get('message')}")
        return res.get("result", {}).get("content")


# Global singleton MCP client
mcp_client = YogaSutraMCPClient()
