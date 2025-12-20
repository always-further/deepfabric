"""Spin Framework HTTP client for tool execution."""

import json
import logging
import uuid

from http import HTTPStatus
from typing import Any

import httpx

from .models import SpinComponentsResponse, SpinExecutionResult, SpinHealthResponse

logger = logging.getLogger(__name__)


class SpinClient:
    """HTTP client for communicating with Spin service."""

    def __init__(
        self,
        endpoint: str,
        timeout: float = 30.0,
        tool_execute_path: str | None = None,
    ):
        """Initialize Spin client.

        Args:
            endpoint: Base URL of Spin service (e.g., "http://localhost:3000")
            timeout: Request timeout in seconds
            tool_execute_path: Custom path for tool execution (e.g., "/mock/execute").
                              When set, uses MCP-style payload format.
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.tool_execute_path = tool_execute_path
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure async client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> SpinHealthResponse:
        """Check if Spin service is healthy.

        Returns:
            SpinHealthResponse with status and available components

        Raises:
            httpx.HTTPError: If service is unreachable
        """
        client = await self._ensure_client()
        response = await client.get(f"{self.endpoint}/health")
        response.raise_for_status()
        return SpinHealthResponse.model_validate(response.json())

    async def get_components(self) -> list[str]:
        """Get list of available tool components.

        Returns:
            List of component names

        Raises:
            httpx.HTTPError: If request fails
        """
        client = await self._ensure_client()
        response = await client.get(f"{self.endpoint}/components")
        response.raise_for_status()
        data = SpinComponentsResponse.model_validate(response.json())
        return data.components

    async def execute_tool(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        component: str | None = None,
    ) -> SpinExecutionResult:
        """Execute a tool via Spin.

        Args:
            session_id: Session identifier for state isolation
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            component: Spin component that implements this tool (e.g., 'vfs', 'github')

        Returns:
            SpinExecutionResult with success status and result/error
        """
        client = await self._ensure_client()

        # Use custom tool_execute_path if configured (MCP/mock style)
        if self.tool_execute_path:
            execute_url = f"{self.endpoint}{self.tool_execute_path}"
            # MCP-style payload format
            payload = {
                "name": tool_name,
                "arguments": arguments,
            }
        else:
            # Standard component-based routing
            if component:
                execute_url = f"{self.endpoint}/{component}/execute"
            else:
                execute_url = f"{self.endpoint}/execute"
            # Standard payload format
            payload = {
                "session_id": session_id,
                "tool": tool_name,
                "args": arguments,
            }

        try:
            response = await client.post(
                execute_url,
                json=payload,
            )

            if response.status_code == HTTPStatus.OK:
                data = response.json()

                # Handle MCP/mock response format (has "result" key with nested data)
                if self.tool_execute_path and "result" in data:
                    # Mock component returns {"result": {...}}

                    return SpinExecutionResult(
                        success=True,
                        result=json.dumps(data["result"]),
                        error_type=None,
                    )

                # Standard Spin component response format
                return SpinExecutionResult(
                    success=data.get("success", False),
                    result=data.get("result", ""),
                    error_type=data.get("error_type"),
                )

            error_data = response.json() if response.content else {}
            return SpinExecutionResult(
                success=False,
                result=error_data.get("error", f"HTTP {response.status_code}"),
                error_type="HTTPError",
            )

        except httpx.TimeoutException:
            return SpinExecutionResult(
                success=False,
                result="Tool execution timed out",
                error_type="Timeout",
            )
        except httpx.HTTPError as e:
            return SpinExecutionResult(
                success=False,
                result=f"HTTP error: {e!s}",
                error_type="HTTPError",
            )

    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up all state for a session.

        Args:
            session_id: Session to clean up

        Returns:
            True if cleanup succeeded
        """
        client = await self._ensure_client()

        try:
            response = await client.delete(f"{self.endpoint}/session/{session_id}")
        except httpx.HTTPError as e:
            logger.warning("Failed to cleanup session %s: %s", session_id, e)
            return False
        else:
            return response.status_code == HTTPStatus.OK


class SpinSession:
    """Manages a Spin session with state persistence across tool calls."""

    def __init__(self, client: SpinClient, session_id: str | None = None):
        """Initialize a Spin session.

        Args:
            client: SpinClient instance
            session_id: Optional session ID (generates UUID if not provided)
        """
        self.client = client
        self.session_id = session_id or str(uuid.uuid4())
        self._initialized = False

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        component: str | None = None,
    ) -> SpinExecutionResult:
        """Execute a tool in this session.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            component: Spin component that implements this tool (e.g., 'vfs', 'github')

        Returns:
            SpinExecutionResult with execution outcome
        """
        return await self.client.execute_tool(
            session_id=self.session_id,
            tool_name=tool_name,
            arguments=arguments,
            component=component,
        )

    async def seed_files(self, files: dict[str, str]) -> bool:
        """Seed initial files into the session's virtual filesystem.

        Args:
            files: Dictionary of file_path -> content

        Returns:
            True if all files were seeded successfully
        """
        for file_path, content in files.items():
            result = await self.execute_tool(
                tool_name="write_file",
                arguments={"file_path": file_path, "content": content},
                component="vfs",  # Builtin VFS tool
            )
            if not result.success:
                logger.error("Failed to seed file %s: %s", file_path, result.result)
                return False

        self._initialized = True
        logger.debug("Seeded %d files for session %s", len(files), self.session_id)
        return True

    async def cleanup(self) -> bool:
        """Clean up this session's state.

        Returns:
            True if cleanup succeeded
        """
        result = await self.client.cleanup_session(self.session_id)
        self._initialized = False
        return result

    async def __aenter__(self) -> "SpinSession":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - cleanup session."""
        await self.cleanup()
