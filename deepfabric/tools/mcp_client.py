"""MCP (Model Context Protocol) client for fetching tools from MCP servers.

Supports both stdio and Streamable HTTP transports as per MCP spec 2025-11-25.
See: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
"""

import contextlib
import json
import logging
import os
import selectors
import shlex
import subprocess  # nosec

from typing import Any, Literal

import httpx
import yaml

from pydantic import BaseModel, Field

from ..exceptions import ConfigurationError
from ..schemas import MCPToolDefinition, ToolDefinition, ToolRegistry

logger = logging.getLogger(__name__)

# MCP Protocol Version
MCP_PROTOCOL_VERSION = "2025-11-25"


class MCPClientInfo(BaseModel):
    """Client information for MCP initialization."""

    name: str = Field(default="deepfabric")
    version: str = Field(default="1.0.0")


class MCPServerInfo(BaseModel):
    """Server information from MCP initialization response."""

    name: str
    version: str = ""


class MCPInitializeResult(BaseModel):
    """Result from MCP initialize request."""

    model_config = {"populate_by_name": True}

    protocol_version: str = Field(alias="protocolVersion")
    capabilities: dict[str, Any] = Field(default_factory=dict)
    server_info: MCPServerInfo | None = Field(default=None, alias="serverInfo")


class MCPToolsListResult(BaseModel):
    """Result from MCP tools/list request."""

    tools: list[dict[str, Any]] = Field(default_factory=list)


def _create_jsonrpc_request(
    method: str, params: dict[str, Any] | None = None, request_id: int = 1
) -> str:
    """Create a JSON-RPC 2.0 request message."""
    request: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        request["params"] = params
    return json.dumps(request, separators=(",", ":"))


def _create_jsonrpc_notification(method: str, params: dict[str, Any] | None = None) -> str:
    """Create a JSON-RPC 2.0 notification (no id, no response expected)."""
    notification: dict[str, Any] = {
        "jsonrpc": "2.0",
        "method": method,
    }
    if params is not None:
        notification["params"] = params
    return json.dumps(notification, separators=(",", ":"))


def _parse_jsonrpc_response(response: str) -> dict[str, Any]:
    """Parse a JSON-RPC 2.0 response."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON-RPC response: {e}") from e

    if "error" in data:
        error = data["error"]
        code = error.get("code", "unknown")
        message = error.get("message", "Unknown error")
        raise ConfigurationError(f"MCP server error [{code}]: {message}")

    return data


def _make_timeout_error(timeout: float) -> ConfigurationError:
    """Create a timeout error."""
    return ConfigurationError(f"Timeout waiting for MCP server response after {timeout}s")


def _make_process_terminated_error(exit_code: int, stderr_output: str) -> ConfigurationError:
    """Create a process terminated error."""
    return ConfigurationError(
        f"MCP server process terminated unexpectedly. Exit code: {exit_code}. "
        f"Stderr: {stderr_output}"
    )


def _make_stdout_closed_error() -> ConfigurationError:
    """Create a stdout closed error."""
    return ConfigurationError("MCP server closed stdout without response")


class StdioMCPClient:
    """MCP client using stdio transport.

    Launches an MCP server as a subprocess and communicates via stdin/stdout.
    """

    def __init__(self, command: str, env: dict[str, str] | None = None):
        """Initialize stdio MCP client.

        Args:
            command: Shell command to launch the MCP server
            env: Optional environment variables to pass to the subprocess
        """
        self.command = command
        self.env = env
        self.process: subprocess.Popen | None = None
        self._request_id = 0

    def _next_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id

    def _send_message(self, message: str) -> None:
        """Send a message to the MCP server via stdin."""
        if self.process is None or self.process.stdin is None:
            raise ConfigurationError("MCP server process not started")

        # Messages are newline-delimited
        self.process.stdin.write(message + "\n")
        self.process.stdin.flush()

    def _receive_message(self, timeout: float = 30.0) -> str:
        """Receive a message from the MCP server via stdout."""
        if self.process is None or self.process.stdout is None:
            raise ConfigurationError("MCP server process not started")

        return self._read_line_with_timeout(timeout)

    def _read_line_with_timeout(self, timeout: float) -> str:
        """Read a line from stdout with timeout handling."""
        if self.process is None or self.process.stdout is None:
            raise ConfigurationError("MCP server process not started")

        sel = selectors.DefaultSelector()
        try:
            sel.register(self.process.stdout, selectors.EVENT_READ)
            events = sel.select(timeout=timeout)
            self._check_timeout(events, timeout)
            line = self.process.stdout.readline()

            if not line:
                self._handle_empty_line()

            return line.strip()
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Error reading from MCP server: {e}") from e
        finally:
            sel.close()

    def _check_timeout(self, events: list, timeout: float) -> None:
        """Check if select timed out and raise appropriate error."""
        if not events:
            raise _make_timeout_error(timeout)

    def _handle_empty_line(self) -> None:
        """Handle case where stdout returns empty line."""
        # Check if process has terminated
        if self.process is not None and self.process.poll() is not None:
            stderr_output = ""
            if self.process.stderr:
                stderr_output = self.process.stderr.read()
            raise _make_process_terminated_error(self.process.returncode, stderr_output)
        raise _make_stdout_closed_error()

    def start(self) -> None:
        """Start the MCP server subprocess."""
        # Merge environment
        process_env = os.environ.copy()
        if self.env:
            process_env.update(self.env)

        # Parse command - handle shell commands properly
        try:
            args = shlex.split(self.command)
        except ValueError as e:
            raise ConfigurationError(f"Invalid command: {e}") from e

        try:
            # Security note: Command execution is intentional here - users explicitly
            # provide MCP server commands via --command flag. We use shlex.split() to
            # safely parse the command string and pass a list to Popen (no shell=True),
            # which prevents shell injection. This is a local CLI tool where the user
            # is trusted to run commands on their own machine.
            self.process = subprocess.Popen(  # nosec  # noqa: S603
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=process_env,
                bufsize=1,  # Line buffered
            )
        except FileNotFoundError as e:
            raise ConfigurationError(f"MCP server command not found: {args[0]}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to start MCP server: {e}") from e

    def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def initialize(self) -> MCPInitializeResult:
        """Send initialize request to the MCP server."""
        request = _create_jsonrpc_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": MCPClientInfo().model_dump(),
            },
            request_id=self._next_id(),
        )

        self._send_message(request)
        response_str = self._receive_message()
        response = _parse_jsonrpc_response(response_str)

        # Send initialized notification
        notification = _create_jsonrpc_notification("notifications/initialized")
        self._send_message(notification)

        return MCPInitializeResult.model_validate(response.get("result", {}))

    def list_tools(self) -> list[dict[str, Any]]:
        """Request the list of available tools from the MCP server."""
        request = _create_jsonrpc_request("tools/list", {}, request_id=self._next_id())

        self._send_message(request)
        response_str = self._receive_message()
        response = _parse_jsonrpc_response(response_str)

        result = MCPToolsListResult.model_validate(response.get("result", {}))
        return result.tools

    def __enter__(self) -> "StdioMCPClient":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class HTTPMCPClient:
    """MCP client using Streamable HTTP transport.

    Communicates with an MCP server over HTTP POST/GET.
    """

    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        """Initialize HTTP MCP client.

        Args:
            endpoint: HTTP endpoint URL for the MCP server
            headers: Optional additional headers (e.g., for authentication)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.headers = headers or {}
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0
        self._client: httpx.Client | None = None

    def _next_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id

    def _get_headers(self) -> dict[str, str]:
        """Get headers for HTTP requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
            **self.headers,
        }
        if self.session_id:
            headers["MCP-Session-Id"] = self.session_id
        return headers

    def _send_request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request to the MCP server."""
        if self._client is None:
            raise ConfigurationError("HTTP client not started")

        request_body = _create_jsonrpc_request(method, params, request_id=self._next_id())

        try:
            response = self._client.post(
                self.endpoint,
                content=request_body,
                headers=self._get_headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()

            # Check for session ID in response headers
            if "MCP-Session-Id" in response.headers:
                self.session_id = response.headers["MCP-Session-Id"]

            # Handle SSE response
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                # Parse SSE format - look for data: lines
                for line in response.text.split("\n"):
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str:
                            return _parse_jsonrpc_response(data_str)
                raise ConfigurationError("No data in SSE response")

            # Regular JSON response
            return _parse_jsonrpc_response(response.text)

        except httpx.HTTPStatusError as e:
            raise ConfigurationError(
                f"HTTP error from MCP server: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise ConfigurationError(f"Failed to connect to MCP server: {e}") from e

    def start(self) -> None:
        """Start the HTTP client."""
        self._client = httpx.Client()

    def stop(self) -> None:
        """Stop the HTTP client and terminate session."""
        if self._client is not None:
            # Send session termination if we have a session
            if self.session_id:
                with contextlib.suppress(Exception):
                    self._client.delete(
                        self.endpoint,
                        headers=self._get_headers(),
                        timeout=5.0,
                    )
            self._client.close()
            self._client = None

    def initialize(self) -> MCPInitializeResult:
        """Send initialize request to the MCP server."""
        response = self._send_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": MCPClientInfo().model_dump(),
            },
        )

        # Send initialized notification (fire and forget for HTTP)
        with contextlib.suppress(Exception):
            notification = _create_jsonrpc_notification("notifications/initialized")
            if self._client:
                self._client.post(
                    self.endpoint,
                    content=notification,
                    headers=self._get_headers(),
                    timeout=5.0,
                )

        return MCPInitializeResult.model_validate(response.get("result", {}))

    def list_tools(self) -> list[dict[str, Any]]:
        """Request the list of available tools from the MCP server."""
        response = self._send_request("tools/list", {})
        result = MCPToolsListResult.model_validate(response.get("result", {}))
        return result.tools

    def __enter__(self) -> "HTTPMCPClient":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def _create_mcp_client(
    transport: Literal["stdio", "http"],
    command: str | None = None,
    endpoint: str | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> StdioMCPClient | HTTPMCPClient:
    """Create an MCP client based on transport type.

    Args:
        transport: Transport type - "stdio" or "http"
        command: Shell command to launch MCP server (required for stdio)
        endpoint: HTTP endpoint URL (required for http)
        env: Environment variables for stdio subprocess
        headers: HTTP headers for authentication etc.
        timeout: Request timeout in seconds

    Returns:
        Configured MCP client (StdioMCPClient or HTTPMCPClient)

    Raises:
        ConfigurationError: If required parameters are missing for the transport type
    """
    if transport == "stdio":
        if not command:
            raise ConfigurationError("command is required for stdio transport")
        return StdioMCPClient(command, env=env)
    if transport == "http":
        if not endpoint:
            raise ConfigurationError("endpoint is required for http transport")
        return HTTPMCPClient(endpoint, headers=headers, timeout=timeout)

    raise ConfigurationError(f"Unknown transport: {transport}")


def fetch_tools_from_mcp(
    transport: Literal["stdio", "http"],
    command: str | None = None,
    endpoint: str | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> ToolRegistry:
    """Fetch tools from an MCP server and convert to DeepFabric ToolRegistry.

    Args:
        transport: Transport type - "stdio" or "http"
        command: Shell command to launch MCP server (required for stdio)
        endpoint: HTTP endpoint URL (required for http)
        env: Environment variables for stdio subprocess
        headers: HTTP headers for authentication etc.
        timeout: Request timeout in seconds

    Returns:
        ToolRegistry containing the converted tools

    Raises:
        ConfigurationError: If transport params are invalid or server communication fails
    """
    client = _create_mcp_client(
        transport=transport,
        command=command,
        endpoint=endpoint,
        env=env,
        headers=headers,
        timeout=timeout,
    )

    with client:
        # Initialize the connection
        init_result = client.initialize()
        logger.info(
            "Connected to MCP server: %s (protocol: %s)",
            init_result.server_info.name if init_result.server_info else "unknown",
            init_result.protocol_version,
        )

        # Fetch tools
        mcp_tools = client.list_tools()
        logger.info("Fetched %d tools from MCP server", len(mcp_tools))

        # Convert to ToolDefinition
        tools = []
        for tool_dict in mcp_tools:
            try:
                mcp_tool = MCPToolDefinition.model_validate(tool_dict)
                tools.append(ToolDefinition.from_mcp(mcp_tool))
            except Exception as e:
                logger.warning("Failed to convert tool '%s': %s", tool_dict.get("name", "?"), e)

        return ToolRegistry(tools=tools)


def save_tools_to_file(
    registry: ToolRegistry,
    output_path: str,
    output_format: Literal["deepfabric", "openai"] = "deepfabric",
) -> None:
    """Save tools to a JSON or YAML file.

    Args:
        registry: ToolRegistry to save
        output_path: Output file path (.json or .yaml/.yml)
        output_format: Output format - "deepfabric" (native) or "openai" (TRL compatible)
    """
    if output_format == "openai":
        data = {"tools": registry.to_openai_format()}
    else:
        data = {"tools": [tool.model_dump() for tool in registry.tools]}

    if output_path.endswith((".yaml", ".yml")):
        with open(output_path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)


class SpinLoadResult(BaseModel):
    """Result from loading tools into Spin."""

    loaded: int = Field(description="The number of tools loaded into Spin.")
    tool_names: list[str] = Field(
        alias="tools", description="The names of the tools loaded into Spin."
    )


def push_tools_to_spin(
    mcp_tools: list[dict[str, Any]],
    spin_endpoint: str,
    timeout: float = 30.0,
) -> SpinLoadResult:
    """Push MCP tools directly to Spin mock component.

    This posts the raw MCP tool definitions to Spin's /mock/load-schema endpoint,
    which accepts MCP format directly.

    Args:
        mcp_tools: List of MCP tool definitions (raw dicts with inputSchema)
        spin_endpoint: Spin server base URL (e.g., "http://localhost:3000")
        timeout: Request timeout in seconds

    Returns:
        SpinLoadResult with count and names of loaded tools

    Raises:
        ConfigurationError: If Spin server is unreachable or returns an error
    """
    endpoint = spin_endpoint.rstrip("/")
    load_url = f"{endpoint}/mock/load-schema"

    # Spin accepts MCP format directly: { "tools": [...] } or just [...]
    payload = {"tools": mcp_tools}

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                load_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

            data = response.json()
            return SpinLoadResult(
                loaded=data.get("loaded", 0),
                tools=data.get("tools", []),
            )

    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_data = e.response.json()
            error_detail = error_data.get("error", "")
        except json.JSONDecodeError:
            error_detail = e.response.text
        raise ConfigurationError(
            f"Spin server returned error {e.response.status_code}: {error_detail}"
        ) from e
    except httpx.ConnectError as e:
        raise ConfigurationError(
            f"Cannot connect to Spin server at {endpoint}. Is it running?"
        ) from e
    except httpx.RequestError as e:
        raise ConfigurationError(f"Failed to communicate with Spin server: {e}") from e


def fetch_and_push_to_spin(
    transport: Literal["stdio", "http"],
    spin_endpoint: str,
    command: str | None = None,
    endpoint: str | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[ToolRegistry, SpinLoadResult]:
    """Fetch tools from MCP server and push to Spin in one operation.

    This is more efficient than fetch + convert + push because it preserves
    the original MCP format (with inputSchema) that Spin expects.

    Args:
        transport: MCP transport type - "stdio" or "http"
        spin_endpoint: Spin server base URL
        command: Shell command for stdio transport
        endpoint: HTTP endpoint for http transport
        env: Environment variables for stdio
        headers: HTTP headers for http transport
        timeout: Request timeout in seconds

    Returns:
        Tuple of (ToolRegistry for local use, SpinLoadResult from Spin)

    Raises:
        ConfigurationError: If either MCP or Spin communication fails
    """
    client = _create_mcp_client(
        transport=transport,
        command=command,
        endpoint=endpoint,
        env=env,
        headers=headers,
        timeout=timeout,
    )

    with client:
        # Initialize the connection
        init_result = client.initialize()
        logger.info(
            "Connected to MCP server: %s (protocol: %s)",
            init_result.server_info.name if init_result.server_info else "unknown",
            init_result.protocol_version,
        )

        # Fetch tools (raw MCP format)
        mcp_tools = client.list_tools()
        logger.info("Fetched %d tools from MCP server", len(mcp_tools))

        # Push raw MCP tools to Spin (preserves inputSchema format)
        spin_result = push_tools_to_spin(mcp_tools, spin_endpoint, timeout=timeout)
        logger.info("Pushed %d tools to Spin server", spin_result.loaded)

        # Also convert to ToolRegistry for local use/saving
        tools = []
        for tool_dict in mcp_tools:
            try:
                mcp_tool = MCPToolDefinition.model_validate(tool_dict)
                tools.append(ToolDefinition.from_mcp(mcp_tool))
            except Exception as e:
                logger.warning("Failed to convert tool '%s': %s", tool_dict.get("name", "?"), e)

        return ToolRegistry(tools=tools), spin_result
