"""Default tool definitions for common use cases."""

from ..schemas import ToolDefinition, ToolParameter, ToolRegistry

# =============================================================================
# Builtin Tools - Execute via Spin VFS component
# These are the only hardcoded tools. Custom components (github, slack, etc.)
# should be configured via YAML config with tools loaded from endpoints.
# =============================================================================

# Component mapping: "builtin" in config routes to "/vfs/execute" in Spin
BUILTIN_COMPONENT = "vfs"

READ_FILE_TOOL = ToolDefinition(
    name="read_file",
    description="Read content from a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to read",
            required=True,
        ),
    ],
    returns="File content as a string",
    category="filesystem",
    component=BUILTIN_COMPONENT,
)

WRITE_FILE_TOOL = ToolDefinition(
    name="write_file",
    description="Write content to a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to write",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="str",
            description="Content to write to the file",
            required=True,
        ),
    ],
    returns="Confirmation message with bytes written",
    category="filesystem",
    component=BUILTIN_COMPONENT,
)

LIST_FILES_TOOL = ToolDefinition(
    name="list_files",
    description="List all files in the current session",
    parameters=[],
    returns="JSON array of file paths",
    category="filesystem",
    component=BUILTIN_COMPONENT,
)

DELETE_FILE_TOOL = ToolDefinition(
    name="delete_file",
    description="Delete a file",
    parameters=[
        ToolParameter(
            name="file_path",
            type="str",
            description="Path to the file to delete",
            required=True,
        ),
    ],
    returns="Confirmation that file was deleted",
    category="filesystem",
    component=BUILTIN_COMPONENT,
)

# Builtin tools registry
BUILTIN_TOOL_REGISTRY = ToolRegistry(
    tools=[
        READ_FILE_TOOL,
        WRITE_FILE_TOOL,
        LIST_FILES_TOOL,
        DELETE_FILE_TOOL,
    ]
)
