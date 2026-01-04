"""API key prompt for notebooks and terminals."""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Cache for API key to avoid repeated prompts
_api_key_cache: str | None = None
_api_key_checked: bool = False


def _is_notebook() -> bool:
    """Check if running in Jupyter/Colab notebook."""
    try:
        from IPython import get_ipython  # type: ignore # noqa: PLC0415

        shell = get_ipython()
        if shell is None:
            is_nb = False
        else:
            shell_name = shell.__class__.__name__
            # ZMQInteractiveShell = Jupyter, Shell = Colab
            is_nb = shell_name in ("ZMQInteractiveShell", "Shell", "Google Colab")
    except (NameError, AttributeError, ImportError):
        return False
    else:
        return is_nb


def _is_colab() -> bool:
    """Check if running in Google Colab specifically."""
    try:
        import google.colab  # type: ignore # noqa: F401, PLC0415
    except ImportError:
        return False
    else:
        return True


def _is_interactive_terminal() -> bool:
    """Check if running in interactive terminal."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _show_notebook_prompt() -> str | None:
    """Show inline widget in Jupyter/Colab (like wandb).

    Returns:
        API key string or None if skipped
    """
    try:
        import ipywidgets as widgets  # type: ignore # noqa: PLC0415

        from IPython.display import HTML, display  # type: ignore # noqa: PLC0415
    except ImportError:
        logger.debug("ipywidgets not available, falling back to terminal prompt")
        return None

    # Result container for callback
    result = {"key": None, "submitted": False}

    # Create styled input widget
    api_key_input = widgets.Password(
        placeholder="Enter your DeepFabric API key",
        description="",
        layout=widgets.Layout(width="300px"),
        style={"description_width": "0px"},
    )

    submit_button = widgets.Button(
        description="Submit",
        button_style="primary",
        layout=widgets.Layout(width="80px"),
    )

    skip_button = widgets.Button(
        description="Skip",
        button_style="",
        layout=widgets.Layout(width="80px"),
        tooltip="Disable logging for this session",
    )

    status_output = widgets.Output()

    def on_submit(_button):  # noqa: ARG001
        key = api_key_input.value.strip()
        if key:
            result["key"] = key
            result["submitted"] = True
            with status_output:
                status_output.clear_output()
                print("API key set. Training metrics will be logged to DeepFabric.")
        else:
            with status_output:
                status_output.clear_output()
                print("Please enter a valid API key.")

    def on_skip(_button):  # noqa: ARG001
        result["key"] = None
        result["submitted"] = True
        with status_output:
            status_output.clear_output()
            print("Logging disabled for this session.")

    submit_button.on_click(on_submit)
    skip_button.on_click(on_skip)

    # Display styled header
    display(
        HTML(
            """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 8px; padding: 16px; margin: 8px 0; color: white;">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                <span style="font-size: 18px; font-weight: 600;">DeepFabric Training Metrics</span>
            </div>
            <p style="margin: 0; font-size: 14px; opacity: 0.9;">
                Enter your API key to automatically log training metrics.<br>
                You can create a key from your profile page: <a href="https://deepfabric.cloud/profile"
                   target="_blank" style="color: #fff; text-decoration: underline;">
                   deepfabric.cloud/profile</a>
            </p>
        </div>
    """
        )
    )

    # Display input widgets
    input_box = widgets.HBox(
        [api_key_input, submit_button, skip_button],
        layout=widgets.Layout(margin="8px 0"),
    )
    display(input_box)
    display(status_output)

    # Wait for user input (with timeout)
    timeout = 300  # 5 minutes
    start = time.monotonic()

    while not result["submitted"] and (time.monotonic() - start) < timeout:
        time.sleep(0.1)

    if not result["submitted"]:
        # Timeout - treat as skip
        with status_output:
            status_output.clear_output()
            print("Prompt timed out. Logging disabled for this session.")
        return None

    return result["key"]


def _show_colab_prompt() -> str | None:
    """Show Colab-specific prompt using getpass.

    Returns:
        API key string or None if skipped
    """
    try:
        from getpass import getpass  # noqa: PLC0415

        from IPython.display import HTML, display  # type: ignore # noqa: PLC0415

        display(
            HTML(
                """
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        border-radius: 8px; padding: 16px; margin: 8px 0; color: white;">
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 8px;">
                    DeepFabric Training Metrics
                </div>
                <p style="margin: 0; font-size: 14px; opacity: 0.9;">
                    Enter your API key below to log training metrics.<br>
                    You can create a key from your profile page: <a href="https://deepfabric.cloud/profile"
                       target="_blank" style="color: #fff;">deepfabric.cloud/profile</a><br>
                    <em>Press Enter without typing to skip.</em>
                </p>
            </div>
        """
            )
        )

        key = getpass("DeepFabric API Key: ").strip()
    except Exception as e:
        logger.debug(f"Colab prompt failed: {e}")
        return None
    else:
        return key if key else None


def _show_terminal_prompt() -> str | None:
    """Show terminal input prompt.

    Returns:
        API key string or None if skipped
    """
    print()
    print("=" * 60)
    print("  DeepFabric Training Metrics")
    print("=" * 60)
    print()
    print("  Enter your API key to log training metrics to DeepFabric.")
    print("  You can create a key from your profile page: https://deepfabric.cloud/profile")
    print()
    print("  Press Enter without typing to skip (disable logging).")
    print()
    print("=" * 60)

    try:
        key = input("  API Key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        print("  Logging disabled for this session.")
        print()
        return None
    else:
        print()
        if key:
            print("  API key set. Training metrics will be logged.")
        else:
            print("  Logging disabled for this session.")
        print()
        return key if key else None


def get_api_key(force_prompt: bool = False) -> str | None:
    """Get API key from environment or prompt user.

    Priority:
    1. DEEPFABRIC_API_KEY environment variable
    2. Cached value from previous prompt
    3. Interactive prompt (notebook widget or terminal input)
    4. None (silently disable logging in non-interactive environments)

    Args:
        force_prompt: If True, prompt even if env var is set

    Returns:
        API key string or None if unavailable/skipped
    """
    global _api_key_cache, _api_key_checked  # noqa: PLW0603

    # Check environment variable first
    env_key = os.getenv("DEEPFABRIC_API_KEY")
    if env_key and not force_prompt:
        return env_key

    # Return cached value if already checked
    if _api_key_checked and not force_prompt:
        return _api_key_cache

    # Mark as checked to avoid repeated prompts
    _api_key_checked = True

    # Try interactive prompts
    if _is_colab():
        try:
            _api_key_cache = _show_colab_prompt()
        except Exception as e:
            logger.debug(f"Colab prompt failed: {e}")
        else:
            return _api_key_cache

    if _is_notebook():
        try:
            _api_key_cache = _show_notebook_prompt()
        except Exception as e:
            logger.debug(f"Notebook prompt failed: {e}")
        else:
            if _api_key_cache is not None:
                return _api_key_cache

    if _is_interactive_terminal():
        try:
            _api_key_cache = _show_terminal_prompt()
        except Exception as e:
            logger.debug(f"Terminal prompt failed: {e}")
        else:
            return _api_key_cache

    # Non-interactive environment - silently disable
    logger.debug("Non-interactive environment, auto-logging disabled")
    _api_key_cache = None
    return None


def clear_api_key_cache() -> None:
    """Clear the cached API key (for testing)."""
    global _api_key_cache, _api_key_checked  # noqa: PLW0603
    _api_key_cache = None
    _api_key_checked = False
