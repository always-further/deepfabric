"""Cloud upload functionality for DeepFabric datasets and topic graphs.

This module provides functions to upload locally generated datasets and topic graphs
to DeepFabric Cloud. It supports both interactive mode (with prompts) and headless
mode for CI/CD pipelines.

Feature is gated behind the EXPERIMENTAL_DF environment variable.
"""

import contextlib
import json
import os
import re
import time

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

    from .tui import DeepFabricTUI

import click
import httpx

from rich.panel import Panel

from .auth import (
    DEFAULT_API_URL,
    clear_tokens,
    device_flow_login,
    get_auth_token,
    get_config,
    is_authenticated,
    save_config,
)
from .tui import get_tui
from .utils import get_bool_env

# HTTP status codes
HTTP_UNAUTHORIZED = 401
HTTP_BAD_REQUEST = 400
HTTP_CONFLICT = 409
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_TOO_MANY_REQUESTS = 429

# Default frontend URL - derived from API URL or explicit env var
DEFAULT_FRONTEND_URL = "https://deepfabric.cloud"


def derive_frontend_url(api_url: str = DEFAULT_API_URL) -> str:
    """Derive the frontend URL from the API URL.

    Args:
        api_url: The API URL (e.g., https://api.deepfabric.cloud)

    Returns:
        The frontend URL (e.g., https://deepfabric.cloud)
    """
    # Check for explicit override first
    explicit_url = os.getenv("DEEPFABRIC_FRONTEND_URL")
    if explicit_url:
        return explicit_url.rstrip("/")

    # Derive from API URL
    if "localhost" in api_url or "127.0.0.1" in api_url:
        # Local development - assume frontend on port 3000
        return "http://localhost:3000"

    if "api." in api_url:
        return api_url.replace("api.", "app.").rstrip("/")

    # Fallback to default
    return DEFAULT_FRONTEND_URL


def get_current_user(api_url: str = DEFAULT_API_URL) -> dict | None:
    """Fetch current user info from /api/v1/auth/me.

    Caches the username in the config file to avoid repeated API calls.

    Args:
        api_url: The DeepFabric API URL

    Returns:
        User info dict with id, email, name, username, or None if failed
    """
    token = get_auth_token()
    if not token:
        return None

    # Check if we have cached user info
    config = get_config()
    cached_username = config.get("username")
    cached_user_id = config.get("user_id")

    if cached_username and cached_user_id:
        return {
            "id": cached_user_id,
            "username": cached_username,
            "email": config.get("email"),
            "name": config.get("name"),
        }

    # Fetch from API
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{api_url}/api/v1/auth/me",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10.0,
            )
            response.raise_for_status()
            user_data = response.json()

            # Cache user info
            config["user_id"] = user_data.get("id")
            config["username"] = user_data.get("username")
            config["email"] = user_data.get("email")
            config["name"] = user_data.get("name")
            save_config(config)

            return user_data
    except (httpx.RequestError, httpx.HTTPStatusError, json.JSONDecodeError, KeyError):
        return None


def derive_name_and_slug(file_path: str) -> tuple[str, str]:
    """Derive name and slug from a file path.

    Args:
        file_path: Path to the file (e.g., "my-dataset.jsonl")

    Returns:
        Tuple of (name, slug) derived from filename
    """
    path = Path(file_path)
    # Remove extension(s) like .jsonl or .json
    stem = path.stem
    if stem.endswith(".json"):
        stem = stem[:-5]

    # Clean up the name for display
    name = stem.replace("-", " ").replace("_", " ").title()

    # Create slug: lowercase, alphanumeric with hyphens
    slug = re.sub(r"[^a-z0-9-]", "-", stem.lower())
    slug = re.sub(r"-+", "-", slug)  # Collapse multiple hyphens
    slug = slug.strip("-")  # Remove leading/trailing hyphens

    # Ensure slug is not empty
    if not slug:
        slug = "unnamed-dataset"
        name = "Unnamed Dataset"

    return name, slug


def prompt_for_name(
    resource_type: str,
    default_name: str,
    default_slug: str,
) -> tuple[str, str]:
    """Prompt user for a name and slug for the resource.

    Args:
        resource_type: Type of resource ("dataset" or "graph")
        default_name: Default name derived from filename
        default_slug: Default slug derived from filename

    Returns:
        Tuple of (name, slug) from user input
    """
    tui = get_tui()
    console = tui.console

    # Visual separator
    console.print()

    # Show a nice header for the naming section
    resource_emoji = "[blue]" if resource_type == "dataset" else "[magenta]"
    console.print(f"  {resource_emoji}{resource_type.upper()}[/] - Enter details for cloud upload")
    console.print()

    # Single prompt for name - slug is auto-derived
    name = click.prompt(
        click.style("    Name", fg="cyan"),
        default=default_name,
        type=str,
    ).strip()

    # Auto-derive slug from name (user doesn't need to worry about this)
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = default_slug

    # Show the URL that will be created
    console.print(f"    [dim]URL path:[/] [cyan]{slug}[/]")
    console.print()

    return name, slug


def upload_dataset(
    dataset_path: str,
    name: str,
    slug: str,
    description: str = "",
    tags: list[str] | None = None,
    organization_id: str | None = None,
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """Upload a dataset to DeepFabric Cloud.

    Args:
        dataset_path: Path to the JSONL dataset file
        name: Display name for the dataset
        slug: URL-friendly slug for the dataset
        description: Optional description
        tags: Optional list of tags
        organization_id: Optional organization UUID
        api_url: The DeepFabric API URL

    Returns:
        Response dict with dataset_id, version_id, and URLs

    Raises:
        Exception: If upload fails
    """
    token = get_auth_token()
    if not token:
        raise ValueError("Not authenticated. Please run 'deepfabric auth login' first.")

    # Read and parse the JSONL file
    samples = []
    with open(dataset_path) as f:
        for line_num, raw_line in enumerate(f, 1):
            content = raw_line.strip()
            if not content:
                continue
            try:
                sample = json.loads(content)
                samples.append(sample)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e

    if not samples:
        raise ValueError("Dataset file is empty or contains no valid samples")

    # Build request payload
    payload = {
        "name": name,
        "slug": slug,
        "description": description,
        "tags": tags or [],
        "samples": samples,
    }
    if organization_id:
        payload["organization_id"] = organization_id

    # Make the upload request
    with httpx.Client() as client:
        response = client.post(
            f"{api_url}/api/v1/datasets/push",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=120.0,  # Allow longer timeout for large uploads
        )
        response.raise_for_status()
        return response.json()


def upload_topic_graph(
    graph_path: str,
    name: str,
    description: str = "",
    slug: str | None = None,
    api_url: str = DEFAULT_API_URL,
) -> dict:
    """Upload a topic graph to DeepFabric Cloud.

    Args:
        graph_path: Path to the JSON graph file
        name: Display name for the graph
        description: Optional description
        slug: Optional URL-friendly slug (derived from name if not provided)
        api_url: The DeepFabric API URL

    Returns:
        Response dict with graph info

    Raises:
        Exception: If upload fails
    """
    token = get_auth_token()
    if not token:
        raise ValueError("Not authenticated. Please run 'deepfabric auth login' first.")

    # Derive slug from name if not provided
    if not slug:
        slug = re.sub(r"[^a-z0-9-]", "-", name.lower())
        slug = re.sub(r"-+", "-", slug).strip("-")

    # Read the graph file
    with open(graph_path) as f:
        graph_content = f.read()

    # Build metadata
    metadata = {
        "name": name,
        "description": description,
        "slug": slug,
    }

    # Make the multipart upload request
    # Use .json extension for the uploaded filename regardless of actual file extension
    # The backend expects .json for graph imports
    upload_filename = Path(graph_path).stem + ".json"

    with httpx.Client() as client:
        response = client.post(
            f"{api_url}/api/v1/topic-graphs/import",
            headers={"Authorization": f"Bearer {token}"},
            files={"file": (upload_filename, graph_content, "application/json")},
            data={"metadata": json.dumps(metadata)},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()


def build_urls(
    resource_type: str,
    resource_id: str,
    slug: str,
    username: str | None,
    frontend_url: str = DEFAULT_FRONTEND_URL,
) -> tuple[str | None, str]:
    """Build user-facing URLs for an uploaded resource.

    Args:
        resource_type: Either "datasets" or "graphs"
        resource_id: UUID of the resource
        slug: URL slug of the resource
        username: Username for namespace URL (None if unavailable)
        frontend_url: Frontend base URL

    Returns:
        Tuple of (public_url, internal_url)
        public_url may be None if username is not available
    """
    internal_url = f"{frontend_url}/{resource_type}/{resource_id}"

    if username:
        if resource_type == "datasets":
            public_url = f"{frontend_url}/dataset/{username}/{slug}"
        else:
            public_url = f"{frontend_url}/graphs/{username}/{slug}"
    else:
        public_url = None

    return public_url, internal_url


def ensure_authenticated(
    api_url: str = DEFAULT_API_URL,
    headless: bool = False,
) -> bool:
    """Ensure the user is authenticated, prompting if necessary.

    Args:
        api_url: The DeepFabric API URL
        headless: If True, don't prompt for login (fail if not authenticated)

    Returns:
        True if authenticated, False otherwise
    """
    if is_authenticated():
        return True

    if headless:
        return False

    tui = get_tui()
    tui.info("You need to authenticate to upload to DeepFabric Cloud.")

    if not click.confirm("Would you like to log in now?", default=True):
        return False

    return device_flow_login(api_url)


def _get_user_friendly_error(e: httpx.HTTPStatusError) -> str:  # noqa: PLR0911
    """Convert HTTP error to user-friendly message.

    Args:
        e: The HTTP status error

    Returns:
        A user-friendly error message
    """
    try:
        error_data = e.response.json()
        raw_message = error_data.get("message", "")

        # Hide database-specific error details from users
        if "unique constraint" in raw_message.lower():
            return "A resource with this name already exists"
        if "foreign key" in raw_message.lower():
            return "Invalid reference to related resource"
        if "sqlstate" in raw_message.lower():
            return "Server error occurred. Please try again."

        # Return cleaned message if it exists
        if raw_message:
            return raw_message

    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback based on status code
    status_code = e.response.status_code
    if status_code == HTTP_BAD_REQUEST:
        return "Invalid request. Please check your input."
    if status_code == HTTP_UNAUTHORIZED:
        return "Authentication required"
    if status_code == HTTP_CONFLICT:
        return "Resource conflict - name may already exist"
    if status_code >= HTTP_INTERNAL_SERVER_ERROR:
        return "Server error occurred. Please try again later."

    return f"Request failed (HTTP {status_code})"


def _is_duplicate_name_error(response: httpx.Response) -> bool:
    """Check if the error is a duplicate name/slug conflict.

    Args:
        response: The HTTP response object

    Returns:
        True if this is a duplicate name error
    """
    if response.status_code not in (HTTP_BAD_REQUEST, HTTP_CONFLICT):
        return False

    try:
        error_data = response.json()
        message = error_data.get("message", "").lower()
        error_field = error_data.get("error", "").lower()
        combined = f"{message} {error_field}"
        # Check for common duplicate/unique constraint error patterns
        return any(
            pattern in combined
            for pattern in [
                "duplicate",
                "unique constraint",
                "already exists",
                "name already",
                "slug already",
                "idx_datasets_owner_slug",
                "idx_topic_graphs_owner_slug",
            ]
        )
    except Exception:
        return False


def _handle_auth_error(
    api_url: str,
    headless: bool,
) -> bool:
    """Handle authentication errors by prompting for re-login.

    Args:
        api_url: The DeepFabric API URL
        headless: If True, don't prompt for re-login

    Returns:
        True if re-authenticated successfully, False otherwise
    """
    tui = get_tui()

    # Clear invalid tokens
    clear_tokens()

    if headless:
        tui.error(
            "Authentication token expired or invalid. "
            "Please run 'deepfabric auth login' to re-authenticate."
        )
        return False

    # Prompt user to re-authenticate
    tui.warning("Your session has expired or is invalid.")
    console = tui.console
    console.print()

    if not click.confirm("  Would you like to log in again?", default=True):
        return False

    console.print()
    return device_flow_login(api_url)


def _display_upload_result(
    tui,
    dataset_result: dict | None,
    graph_result: dict | None,
    username: str | None,
    frontend_url: str = DEFAULT_FRONTEND_URL,
) -> dict:
    """Display upload results in a formatted panel.

    Args:
        tui: TUI instance for output
        dataset_result: Result from dataset upload (or None)
        graph_result: Result from graph upload (or None)
        username: Username for namespace URLs
        frontend_url: Frontend base URL

    Returns:
        Dict with all URLs for JSON output
    """
    result: dict[str, str | bool] = {"success": True}
    lines: list[str] = []

    if dataset_result:
        dataset_id = dataset_result.get("dataset_id", "")
        # Get slug from the result or derive it
        slug = dataset_result.get("slug", dataset_result.get("name", "").lower().replace(" ", "-"))
        public_url, internal_url = build_urls("datasets", dataset_id, slug, username, frontend_url)

        result["dataset_id"] = dataset_id
        result["dataset_internal_url"] = internal_url
        if public_url:
            result["dataset_url"] = public_url
            lines.append("[bold blue]Dataset[/bold blue]")
            lines.append(f"  [cyan]{public_url}[/cyan]")
            lines.append(f"  [dim]{internal_url}[/dim]")
        else:
            lines.append("[bold blue]Dataset[/bold blue]")
            lines.append(f"  [cyan]{internal_url}[/cyan]")

    if graph_result:
        graph_info = graph_result.get("graph", {})
        graph_id = graph_info.get("id", "")
        slug = graph_info.get("slug", graph_info.get("name", "").lower().replace(" ", "-"))
        public_url, internal_url = build_urls("graphs", graph_id, slug, username, frontend_url)

        result["graph_id"] = graph_id
        result["graph_internal_url"] = internal_url
        if lines:
            lines.append("")  # Add spacing between dataset and graph
        if public_url:
            result["graph_url"] = public_url
            lines.append("[bold magenta]Graph[/bold magenta]")
            lines.append(f"  [cyan]{public_url}[/cyan]")
            lines.append(f"  [dim]{internal_url}[/dim]")
        else:
            lines.append("[bold magenta]Graph[/bold magenta]")
            lines.append(f"  [cyan]{internal_url}[/cyan]")

    if not dataset_result and not graph_result:
        lines.append("[dim](no uploads)[/dim]")

    # Display the panel with nice formatting
    console = tui.console
    console.print()
    panel_content = "\n".join(lines)
    console.print(
        Panel(
            panel_content,
            title="[bold green]Uploaded to DeepFabric Cloud[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    return result


def _upload_with_retry(
    upload_fn,
    max_retries: int = 3,
    initial_delay: float = 1.0,
) -> dict:
    """Execute an upload function with exponential backoff retry.

    Args:
        upload_fn: Function to call that performs the upload
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Result from the upload function

    Raises:
        Exception: If all retries fail
    """
    last_error: Exception | None = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return upload_fn()
        except httpx.HTTPStatusError as e:
            last_error = e
            # Don't retry client errors (4xx) except rate limits
            status_code = e.response.status_code
            is_client_error = HTTP_BAD_REQUEST <= status_code < HTTP_INTERNAL_SERVER_ERROR
            if is_client_error and status_code != HTTP_TOO_MANY_REQUESTS:
                raise

            if attempt < max_retries:
                # Check for Retry-After header
                retry_after = e.response.headers.get("Retry-After")
                if retry_after:
                    with contextlib.suppress(ValueError):
                        delay = float(retry_after)

                time.sleep(delay)
                delay *= 2  # Exponential backoff
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2

    if last_error is not None:
        raise last_error
    # This should never happen as the loop always catches an exception
    raise RuntimeError("Upload failed with unknown error")


def _perform_upload(
    resource_type: str,
    file_path: str,
    upload_fn: Callable[[str, str], dict],
    success_message_fn: Callable[[dict], str],
    tui: "DeepFabricTUI",
    console: "Console",
    api_url: str,
    headless: bool,
) -> dict | None:
    """Perform upload with retry logic for auth errors and duplicate names.

    Args:
        resource_type: Type of resource ("dataset" or "graph")
        file_path: Path to the file to upload
        upload_fn: Function that takes (name, slug) and performs the upload
        success_message_fn: Function that takes the result and returns a success message
        tui: TUI instance for displaying messages
        console: Rich console instance
        api_url: API URL for re-authentication
        headless: Whether running in headless mode

    Returns:
        Upload result dict or None if upload failed/skipped
    """
    default_name, default_slug = derive_name_and_slug(file_path)
    if headless:
        name, slug = default_name, default_slug
    else:
        name, slug = prompt_for_name(resource_type, default_name, default_slug)

    # Loop to handle retries (auth errors, duplicate names)
    max_name_retries = 3
    for _attempt in range(max_name_retries):
        tui.info(f"Uploading {resource_type} '{name}'...")

        try:
            result = _upload_with_retry(lambda n=name, s=slug: upload_fn(n, s))
        except httpx.HTTPStatusError as e:
            # Check for auth error (401)
            if e.response.status_code == HTTP_UNAUTHORIZED:
                if _handle_auth_error(api_url, headless):
                    # Re-authenticated, retry upload
                    continue
                # User declined re-auth or headless mode
                if headless:
                    raise click.ClickException(
                        f"{resource_type.capitalize()} upload failed: authentication required"
                    ) from None
                return None

            # Check for duplicate name error
            if _is_duplicate_name_error(e.response) and not headless:
                tui.warning(
                    f"A {resource_type} named '{slug}' already exists. "
                    "Please choose a different name."
                )
                console.print()
                name, slug = prompt_for_name(resource_type, name + "-2", slug + "-2")
                continue

            # Other HTTP errors - show user-friendly message
            error_msg = _get_user_friendly_error(e)
            tui.error(f"Failed to upload {resource_type}: {error_msg}")
            if headless:
                raise click.ClickException(
                    f"{resource_type.capitalize()} upload failed: {error_msg}"
                ) from None
            return None
        except Exception as e:
            tui.error(f"Failed to upload {resource_type}: {e}")
            if headless:
                raise click.ClickException(
                    f"{resource_type.capitalize()} upload failed: {e}"
                ) from None
            return None
        else:
            tui.success(success_message_fn(result))
            return result

    return None


def handle_cloud_upload(  # noqa: PLR0911
    dataset_path: str | None = None,
    graph_path: str | None = None,
    cloud_upload_flag: str | None = None,
    api_url: str = DEFAULT_API_URL,
) -> dict | None:
    """Main entry point for cloud upload functionality.

    Args:
        dataset_path: Path to dataset JSONL file (or None)
        graph_path: Path to topic graph JSON file (or None)
        cloud_upload_flag: Upload mode for headless operation:
            - None: Interactive mode with prompts
            - "all": Upload both dataset and graph
            - "dataset": Upload dataset only
            - "graph": Upload graph only
            - "none": Skip uploads
        api_url: The DeepFabric API URL

    Returns:
        Dict with upload results and URLs, or None if skipped
    """
    # Check experimental flag
    if not get_bool_env("EXPERIMENTAL_DF"):
        return None

    tui = get_tui()
    console = tui.console
    headless = cloud_upload_flag is not None

    # Derive frontend URL from API URL
    frontend_url = derive_frontend_url(api_url)

    # Determine what to upload
    if headless:
        # Headless mode - use flag value directly
        if cloud_upload_flag == "none":
            return None

        upload_dataset_flag = cloud_upload_flag in ("all", "dataset")
        upload_graph_flag = cloud_upload_flag in ("all", "graph")
    else:
        # Interactive mode - prompt user
        has_dataset = dataset_path and Path(dataset_path).exists()
        has_graph = graph_path and Path(graph_path).exists()

        if not has_dataset and not has_graph:
            return None

        # Visual separator and header
        console.print()
        console.rule("[bold cyan]Cloud Upload[/bold cyan]", style="cyan")
        console.print()

        # Build prompt based on what's available
        if has_dataset and has_graph:
            prompt_text = "  Upload to DeepFabric Cloud?"
            hint = "[dim](Y=both, n=skip, c=choose)[/dim]"
        elif has_dataset:
            prompt_text = "  Upload dataset to DeepFabric Cloud?"
            hint = "[dim](Y=yes, n=skip)[/dim]"
        else:
            prompt_text = "  Upload graph to DeepFabric Cloud?"
            hint = "[dim](Y=yes, n=skip)[/dim]"

        console.print(f"{prompt_text} {hint}")

        if has_dataset and has_graph:
            response = click.prompt(
                click.style("  Choice", fg="cyan"),
                type=click.Choice(["Y", "n", "c"], case_sensitive=False),
                default="Y",
                show_choices=False,
            )
        else:
            response = click.prompt(
                click.style("  Choice", fg="cyan"),
                type=click.Choice(["Y", "n"], case_sensitive=False),
                default="Y",
                show_choices=False,
            )

        if response.lower() == "n":
            tui.info("Skipping cloud upload.")
            return None

        if response.lower() == "c":
            # Choose individually
            upload_dataset_flag = has_dataset and click.confirm("Upload dataset?", default=True)
            upload_graph_flag = has_graph and click.confirm("Upload graph?", default=True)
        else:
            # Y = upload all available
            upload_dataset_flag = has_dataset
            upload_graph_flag = has_graph

    # Check if anything to upload
    if not upload_dataset_flag and not upload_graph_flag:
        return None

    # Ensure authenticated
    if not ensure_authenticated(api_url, headless=headless):
        if headless:
            tui.error(
                "Authentication required. Set DEEPFABRIC_API_KEY environment variable "
                "or run 'deepfabric auth login' first."
            )
            raise click.ClickException("Authentication required for cloud upload")
        tui.info("Skipping cloud upload (not authenticated).")
        return None

    # Get user info for URL construction
    user_info = get_current_user(api_url)
    username = user_info.get("username") if user_info else None

    # Upload dataset
    dataset_result = None
    if upload_dataset_flag and dataset_path:
        dataset_result = _perform_upload(
            resource_type="dataset",
            file_path=dataset_path,
            upload_fn=lambda n, s: upload_dataset(
                dataset_path=dataset_path, name=n, slug=s, api_url=api_url
            ),
            success_message_fn=lambda r: f"Dataset uploaded: {r.get('samples_count', 0)} samples",
            tui=tui,
            console=console,
            api_url=api_url,
            headless=headless,
        )

    # Upload graph
    graph_result = None
    if upload_graph_flag and graph_path:
        graph_result = _perform_upload(
            resource_type="graph",
            file_path=graph_path,
            upload_fn=lambda n, s: upload_topic_graph(
                graph_path=graph_path, name=n, slug=s, api_url=api_url
            ),
            success_message_fn=lambda r: (
                f"Graph uploaded: {r.get('nodes_imported', 0)} nodes, "
                f"{r.get('edges_imported', 0)} edges"
            ),
            tui=tui,
            console=console,
            api_url=api_url,
            headless=headless,
        )

    # Display results
    if dataset_result or graph_result:
        result = _display_upload_result(tui, dataset_result, graph_result, username, frontend_url)

        # In headless mode, also output JSON
        if headless:
            tui.console.print(json.dumps(result, indent=2))

        return result

    return None
