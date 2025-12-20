"""Tool loading and management utilities."""

import json
import logging

from typing import Any

import httpx

from pydantic import ValidationError

from ..exceptions import ConfigurationError
from ..schemas import MCPToolDefinition, ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)


def load_tools_from_endpoint(endpoint_url: str, timeout: float = 30.0) -> ToolRegistry:
    """Load tool definitions from an HTTP endpoint in MCP format.

    Fetches tools from an endpoint like /mock/list-tools that returns
    MCP-format tool definitions with inputSchema.

    Args:
        endpoint_url: Full URL to fetch tools from (e.g., 'http://localhost:3000/mock/list-tools')
        timeout: Request timeout in seconds

    Returns:
        ToolRegistry with loaded tools converted from MCP format

    Raises:
        ConfigurationError: If endpoint cannot be reached or returns invalid data
    """
    try:
        response = httpx.get(endpoint_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except httpx.RequestError as e:
        raise ConfigurationError(f"Failed to connect to tools endpoint {endpoint_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        raise ConfigurationError(
            f"Tools endpoint returned error {e.response.status_code}: {e.response.text}"
        ) from e
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON from tools endpoint: {e}") from e

    # Extract tools array - handle both {"tools": [...]} and direct array
    if isinstance(data, dict) and "tools" in data:
        tools_data = data["tools"]
    elif isinstance(data, list):
        tools_data = data
    else:
        raise ConfigurationError(
            f"Invalid response from {endpoint_url}: expected 'tools' key or array"
        )

    # Convert MCP tools to ToolDefinition
    try:
        tools = []
        for tool_dict in tools_data:
            mcp_tool = MCPToolDefinition.model_validate(tool_dict)
            tools.append(ToolDefinition.from_mcp(mcp_tool))

        logger.info("Loaded %d tools from endpoint %s", len(tools), endpoint_url)
        return ToolRegistry(tools=tools)

    except ValidationError as e:
        raise ConfigurationError(f"Invalid MCP tool schema from {endpoint_url}: {e}") from e


def load_tools_from_dict(tool_dicts: list[dict[str, Any]]) -> ToolRegistry:
    """Load tool definitions from a list of dictionaries.

    Args:
        tool_dicts: List of tool definition dictionaries

    Returns:
        ToolRegistry with loaded tools

    Raises:
        ConfigurationError: If tool definitions are invalid
    """
    try:
        tools = [ToolDefinition.model_validate(tool_dict) for tool_dict in tool_dicts]
        return ToolRegistry(tools=tools)
    except Exception as e:
        raise ConfigurationError(f"Invalid tool definitions: {str(e)}") from e
