import contextlib
import logging
import os
import uuid

from pathlib import Path

from posthog import Posthog, identify_context, new_context

from .tui import get_tui

try:
    import importlib.metadata

    VERSION = importlib.metadata.version("deepfabric")
except (ImportError, importlib.metadata.PackageNotFoundError):
    VERSION = "development"


# Initialize PostHog client
posthog = Posthog(
    project_api_key="phc_JZWiTzIDNnBp6Jj6uUb0JQKuIp3dv0gkay9aU50n38h",
    host="https://eu.i.posthog.com",
)

logger = logging.getLogger(__name__)


class _TelemetryState:
    """Holds the state for the telemetry module."""

    def __init__(self) -> None:
        self.debug_trace: bool = False
        self.user_id_announced: bool = False
        self.user_id_cache: str | None = None
        self.telemetry_failed_once: bool = False


_state = _TelemetryState()


APP_NAME = "DeepFabric"
APP_AUTHOR = "DeepFabric"


try:
    from platformdirs import user_data_dir
except ImportError:  # pragma: no cover - optional dependency

    def user_data_dir(appname: str, appauthor: str | None = None) -> str:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or os.path.expanduser(r"~\AppData\Roaming")
        elif os.name == "posix":
            base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
        else:
            base = os.path.expanduser("~")
        return str(Path(base) / (appauthor or appname) / appname)


def _user_id_path() -> Path:
    candidates: list[Path] = []
    try:
        candidates.append(Path(user_data_dir(APP_NAME, APP_AUTHOR)))
    except Exception:
        logger.debug("Failed to resolve platform data dir", exc_info=True)
    candidates.append(Path.home() / f".{APP_NAME.lower()}")
    candidates.append(Path.cwd())

    for data_dir in candidates:
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir / "telemetry_id"
        except Exception:
            logger.debug("Failed to prepare telemetry directory %s", data_dir, exc_info=True)

    return Path("telemetry_id")


def _read_user_id(path: Path) -> str | None:
    try:
        if path.exists():
            candidate = path.read_text(encoding="utf-8").strip()
            if candidate:
                uuid.UUID(candidate)
                return candidate
    except Exception:
        logger.debug("Failed to read existing telemetry id", exc_info=True)
    return None


def _write_user_id(path: Path) -> str:
    user_id = str(uuid.uuid4())
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(user_id, encoding="utf-8")
        if os.name == "posix":
            os.chmod(tmp_path, 0o600)
        tmp_path.replace(path)
    except Exception:
        logger.debug("Failed to persist telemetry id", exc_info=True)
        return user_id
    else:
        return user_id
    finally:
        with contextlib.suppress(Exception):
            if tmp_path.exists() and tmp_path != path:
                tmp_path.unlink()


def _get_user_id() -> str:
    """Generate a stable, anonymous user ID persisted on disk."""
    if _state.user_id_cache is not None:
        return _state.user_id_cache

    path = _user_id_path()
    user_id = _read_user_id(path)
    if user_id is None:
        user_id = _write_user_id(path)

    _state.user_id_cache = user_id
    return user_id


def _is_developer() -> bool:
    """
    Check if this session is marked as a developer session.

    Returns:
        bool: True if DEEPFABRIC_DEVELOPER environment variable is set to 'True'
    """
    return os.environ.get("DEEPFABRIC_DEVELOPER") == "True"


def set_trace_debug(enabled: bool) -> None:
    """Enable or disable debug output for telemetry events."""
    _state.debug_trace = enabled
    if not enabled:
        _state.user_id_announced = False


def _announce_user_id(user_id: str) -> None:
    if _state.user_id_announced or not _state.debug_trace:
        return
    try:
        get_tui().info(f"metrics user id: {user_id}")
    except Exception:  # pragma: no cover - fallback to logging
        logger.debug("metrics user id: %s", user_id)

    _state.user_id_announced = True


def trace(event_name, event_properties=None):
    """
    Send an analytics event if metrics is enabled.

    Uses privacy-respecting identity tracking with a stable, anonymous user ID
    stored on disk for reuse. Developer sessions are marked with
    the is_developer flag when DEEPFABRIC_DEVELOPER=True.

    Args:
        event_name: Name of the event to track
        event_properties: Optional dictionary of event properties
    """
    if not is_enabled():
        return

    try:
        # Generate stable user ID
        user_id = _get_user_id()
        _announce_user_id(user_id)

        # Add version and developer flag to all events
        properties = event_properties or {}
        properties["version"] = VERSION
        properties["is_developer"] = _is_developer()

        # Use identity context to associate events with the user
        with new_context():
            identify_context(user_id)
            posthog.capture(
                distinct_id=user_id,
                event=event_name,
                properties=properties,
            )
    except Exception:
        if not _state.telemetry_failed_once:
            _state.telemetry_failed_once = True
            logger.warning(
                "Failed to send telemetry event. Further failures will be logged at DEBUG level.",
                exc_info=True,
            )
        else:
            logger.debug("Failed to capture metrics event", exc_info=True)


def is_enabled():
    """Check if analytics is currently enabled."""
    return (
        os.environ.get("ANONYMIZED_TELEMETRY") != "False"
        and os.environ.get("DEEPFABRIC_TESTING") != "True"
    )


def shutdown() -> None:
    """
    Shutdown the PostHog client, flushing any buffered events.

    This should be called on application exit to ensure all metrics data is sent.
    """
    if is_enabled():
        logger.debug("Shutting down metrics client.")
        posthog.shutdown()
