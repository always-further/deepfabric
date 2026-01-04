import contextlib
import json
import os
import time
import webbrowser

from pathlib import Path

import click
import httpx

from .tui import get_tui
from .utils import get_bool_env

DEFAULT_API_URL = os.getenv("DEEPFABRIC_API_URL", "https://api.deepfabric.cloud")
DEFAULT_FRONTEND_URL = os.getenv("DEEPFABRIC_FRONTEND_URL", "https://deepfabric.cloud")

CONFIG_DIR = Path.home() / ".deepfabric"
CONFIG_FILE = CONFIG_DIR / "config.json"


def get_config() -> dict:
    """Load authentication config from disk."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict) -> None:
    """Save authentication config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    # Make config file readable only by owner
    CONFIG_FILE.chmod(0o600)


def get_stored_token() -> str | None:
    """Get stored access token."""
    config = get_config()
    return config.get("access_token")


def store_tokens(access_token: str, refresh_token: str) -> None:
    """Store access and refresh tokens."""
    config = get_config()
    config["access_token"] = access_token
    config["refresh_token"] = refresh_token
    save_config(config)


def clear_tokens() -> None:
    """Clear stored tokens."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


def is_authenticated() -> bool:
    """Check if user is authenticated via API key or OAuth token."""
    return bool(get_auth_token())


def get_auth_token() -> str | None:
    """Get authentication token (API key from env, or stored API key/access token)."""
    # Check environment variable first
    env_api_key = os.getenv("DEEPFABRIC_API_KEY")
    if env_api_key:
        return env_api_key
    # Then check stored config
    config = get_config()
    return config.get("api_key") or config.get("access_token")


def prompt_cloud_signup(api_url: str = DEFAULT_API_URL) -> bool:
    """
    Prompt user to login/signup for cloud features.

    Returns:
        True if user successfully authenticated, False otherwise
    """
    if not get_bool_env("EXPERIMENTAL_DF"):
        return False

    tui = get_tui()

    tui.console.print("")
    tui.info("DeepFabric Cloud can save and track your evaluations")
    tui.info("   - Compare models across runs")
    tui.info("   - Track performance over time")
    tui.info("   - Share results with your team")
    tui.console.print("")

    response = click.prompt(
        "Would you like to record this evaluation in DeepFabric Cloud?",
        type=click.Choice(["y", "n"], case_sensitive=False),
        default="y",
        show_choices=True,
    )

    if response.lower() != "y":
        tui.info("Skipping cloud reporting. Results will be saved locally only.")
        return False

    tui.console.print("")
    tui.info("Great! Let's get you authenticated.")
    tui.console.print("")

    auth_choice = click.prompt(
        "Choose authentication method",
        type=click.Choice(["login", "register", "skip"], case_sensitive=False),
        default="login",
        show_choices=True,
    )

    if auth_choice == "skip":
        tui.info("Skipping authentication. Results will be saved locally only.")
        return False

    if auth_choice == "register":
        tui.info("Opening DeepFabric Cloud registration page...")
        register_url = DEFAULT_FRONTEND_URL.rstrip("/") + "/signup"
        with contextlib.suppress(Exception):
            webbrowser.open(register_url)
        tui.info("After registering, come back here to log in.")
        tui.console.print("")

        if not click.confirm("Ready to log in?", default=True):
            tui.info("Skipping authentication. Results will be saved locally only.")
            return False

    # Proceed with login
    success = device_flow_login(api_url)
    if success:
        tui.success("Authentication successful! Your evaluation will be saved to the cloud.")
        return True
    tui.warning("Authentication failed. Results will be saved locally only.")
    return False


def device_flow_login(api_url: str = DEFAULT_API_URL, debug: bool = False) -> bool:  # noqa: PLR0911
    """
    Perform device flow OAuth login.

    Args:
        api_url: The DeepFabric API URL
        debug: If True, print debug information during polling

    Returns:
        True if login successful, False otherwise
    """
    tui = get_tui()

    try:
        # Request device code
        tui.info("Initiating device authorization...")
        with httpx.Client() as client:
            response = client.post(
                f"{api_url}/api/v1/oauth/device/code",
                json={"client_id": "deepfabric-cli"},
                timeout=10.0,
            )
            response.raise_for_status()
            device_data = response.json()

        device_code = device_data["device_code"]
        user_code = device_data["user_code"]
        verification_uri = device_data["verification_uri_complete"]
        expires_in = device_data["expires_in"]
        interval = device_data.get("interval", 5)

        tui.console.print("")
        tui.success(f"Your user code is: {user_code}")
        tui.info(f"Opening browser to: {verification_uri}")
        tui.info("If the browser doesn't open, please visit the URL above manually.")
        tui.console.print("")

        if debug:
            tui.console.print(f"[dim]Debug: device_code={device_code[:20]}...[/dim]")
            tui.console.print(f"[dim]Debug: expires_in={expires_in}s, interval={interval}s[/dim]")

        # Open browser
        with contextlib.suppress(Exception):
            webbrowser.open(verification_uri)

        # Poll for authorization
        start_time = time.time()
        tui.info("Waiting for authorization...")
        poll_count = 0

        with httpx.Client() as client:
            while time.time() - start_time < expires_in:
                time.sleep(interval)
                poll_count += 1

                try:
                    response = client.post(
                        f"{api_url}/api/v1/oauth/device/token",
                        json={
                            "device_code": device_code,
                            "client_id": "deepfabric-cli",
                        },
                        timeout=10.0,
                    )
                    token_data = response.json()

                    if debug:
                        # Mask sensitive data in debug output
                        debug_data = {
                            k: (
                                v[:20] + "..."
                                if k in ("access_token", "refresh_token") and v
                                else v
                            )
                            for k, v in token_data.items()
                        }
                        tui.console.print(f"[dim]Debug poll #{poll_count}: {debug_data}[/dim]")

                    # Check for error responses (OAuth 2.0 device flow standard)
                    error = token_data.get("error")
                    if error == "authorization_pending":
                        continue
                    if error == "slow_down":
                        interval += 5
                        if debug:
                            tui.console.print(
                                f"[dim]Debug: slow_down, new interval={interval}s[/dim]"
                            )
                        continue
                    if error == "expired_token":
                        tui.error("Device code expired. Please try again.")
                        return False
                    if error == "access_denied":
                        tui.error("Access denied by user.")
                        return False
                    if error:
                        tui.error(f"Authorization failed: {error}")
                        return False

                    # Success - check for tokens
                    access_token = token_data.get("access_token")
                    refresh_token = token_data.get("refresh_token")

                    if access_token and refresh_token:
                        store_tokens(access_token, refresh_token)
                        tui.success("Login successful!")
                        return True
                    if access_token:
                        # Some flows may only return access token
                        store_tokens(access_token, "")
                        tui.success("Login successful!")
                        return True

                    # No error and no tokens - unexpected response
                    if debug:
                        tui.console.print("[dim]Debug: No error and no tokens in response[/dim]")

                except httpx.HTTPStatusError as e:
                    # Non-2xx response - check if it contains OAuth error
                    if debug:
                        tui.console.print(f"[dim]Debug: HTTP {e.response.status_code}[/dim]")
                    try:
                        error_data = e.response.json()
                        error = error_data.get("error")
                        if error == "authorization_pending":
                            continue
                        if error == "slow_down":
                            interval += 5
                            continue
                    except Exception:
                        tui.error("Unexpected response during polling.")
                        pass
                    tui.error(f"Request failed: {e}")
                    return False
                except httpx.HTTPError as e:
                    tui.error(f"Network error: {e}")
                    return False

            tui.error("Authorization timed out. Please try again.")
            return False

    except Exception as e:
        tui.error(f"Login failed: {e}")
        return False


@click.group()
def auth():
    """Authentication commands for DeepFabric Cloud."""
    pass


@auth.command()
@click.option(
    "--api-key",
    help="Authenticate using an API key instead of device flow",
)
@click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    help=f"DeepFabric API URL (default: {DEFAULT_API_URL})",
)
@click.pass_context
def login(ctx: click.Context, api_key: str | None, api_url: str):
    """
    Log in to DeepFabric Cloud.

    By default, uses OAuth device flow (opens browser).
    Use --api-key to authenticate with an API key instead.
    """
    tui = get_tui()
    debug = ctx.obj.get("debug", False) if ctx.obj else False

    if api_key:
        # API key authentication
        config = get_config()
        config["api_key"] = api_key
        config["api_url"] = api_url
        save_config(config)
        tui.success("API key saved successfully!")
        tui.info("You can now use DeepFabric Cloud features.")
    else:
        # Device flow authentication
        success = device_flow_login(api_url, debug=debug)
        if success:
            tui.info("You can now use DeepFabric Cloud features.")
        else:
            raise click.ClickException("Login failed")


@auth.command()
def logout():
    """Log out from DeepFabric Cloud."""
    tui = get_tui()

    clear_tokens()
    tui.success("Logged out successfully!")


@auth.command()
@click.option(
    "--api-url",
    default=DEFAULT_API_URL,
    help=f"DeepFabric API URL (default: {DEFAULT_API_URL})",
)
def status(api_url: str):
    """Check authentication status."""
    tui = get_tui()

    config = get_config()
    api_key = config.get("api_key")
    access_token = get_stored_token()

    if not api_key and not access_token:
        tui.warning("Not logged in")
        tui.info("Run 'deepfabric auth login' to authenticate")
        return

    # Try to verify authentication with API
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        with httpx.Client() as client:
            response = client.get(
                f"{api_url}/api/v1/auth/me",
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()
            user_data = response.json()

        tui.success("Authenticated")
        tui.info(f"Email: {user_data['email']}")
        tui.info(f"Name: {user_data.get('name', 'N/A')}")
        tui.info(f"User ID: {user_data['id']}")

        if api_key:
            tui.info("Auth method: API Key")
        else:
            tui.info("Auth method: OAuth")

    except Exception as e:
        tui.error(f"Authentication verification failed: {e}")
        tui.warning("Your stored credentials may be invalid or expired")
        tui.info("Run 'deepfabric auth login' to re-authenticate")
