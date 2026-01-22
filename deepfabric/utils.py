import ast
import asyncio
import importlib
import json
import os
import re

from typing import Any

VALIDATION_ERROR_INDICATORS = [
    "validation error",
    "value error",
    "is null",
    "is empty string",
    "must provide actual value",
    "invalid schema",
    "pydantic",
    "string should have at least",
    "field required",
]


def is_validation_error(error: Exception) -> bool:
    """Check if an error is a validation/schema error that can be retried."""
    error_str = str(error).lower()
    return any(indicator in error_str for indicator in VALIDATION_ERROR_INDICATORS)


def ensure_not_running_loop(method_name: str) -> None:
    """Raise when invoked inside an active asyncio event loop."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    if loop.is_running():
        msg = (
            f"{method_name} cannot be called while an event loop is running. "
            "Use the async variant instead."
        )
        raise RuntimeError(msg)


def extract_list(input_string: str):
    """
    Extracts a Python list from a given input string.

    This function attempts to parse the input string as JSON. If that fails,
    it searches for the first Python list within the string by identifying
    the opening and closing brackets. If a list is found, it is evaluated
    safely to ensure it is a valid Python list.

    Args:
        input_string (str): The input string potentially containing a Python list.

    Returns:
        list: The extracted Python list if found and valid, otherwise an empty list.

    Raises:
        None: This function handles its own exceptions and does not raise any.
    """
    try:
        return json.loads(input_string)
    except json.JSONDecodeError:
        print("Failed to parse the input string as JSON.")

    start = input_string.find("[")
    if start == -1:
        print("No Python list found in the input string.")
        return []

    count = 0
    for i, char in enumerate(input_string[start:]):
        if char == "[":
            count += 1
        elif char == "]":
            count -= 1
        if count == 0:
            end = i + start + 1
            break
    else:
        print("No matching closing bracket found.")
        return []

    found_list_str = input_string[start:end]
    found_list = safe_literal_eval(found_list_str)
    if found_list is None:
        print("Failed to parse the list due to syntax issues.")
        return []

    return found_list


def remove_linebreaks_and_spaces(input_string):
    """
    Remove line breaks and extra spaces from the input string.

    This function replaces all whitespace characters (including line breaks)
    with a single space and then ensures that there are no consecutive spaces
    in the resulting string.

    Args:
        input_string (str): The string from which to remove line breaks and extra spaces.

    Returns:
        str: The processed string with line breaks and extra spaces removed.
    """
    no_linebreaks = re.sub(r"\s+", " ", input_string)
    return " ".join(no_linebreaks.split())


def safe_literal_eval(list_string: str):
    """
    Safely evaluate a string containing a Python literal expression.

    This function attempts to evaluate a string containing a Python literal
    expression using `ast.literal_eval`. If a `SyntaxError` or `ValueError`
    occurs, it tries to sanitize the string by replacing problematic apostrophes
    with the actual right single quote character and attempts the evaluation again.

    Args:
        list_string (str): The string to be evaluated.

    Returns:
        The result of the evaluated string if successful, otherwise `None`.
    """
    try:
        return ast.literal_eval(list_string)
    except (SyntaxError, ValueError):
        # Replace problematic apostrophes with the actual right single quote character
        sanitized_string = re.sub(r"(\w)'(\w)", r"\1’\2", list_string)
        try:
            return ast.literal_eval(sanitized_string)
        except (SyntaxError, ValueError):
            print("Failed to parse the list due to syntax issues.")
            return None


def read_topic_tree_from_jsonl(file_path: str) -> list[dict]:
    """
    Read the topic tree from a JSONL file.

    Args:
        file_path (str): The path to the JSONL file.

    Returns:
        list[dict]: The topic tree.
    """
    topic_tree = []
    with open(file_path) as file:
        for line in file:
            topic_tree.append(json.loads(line.strip()))

    return topic_tree


def parse_num_samples(value: int | str | None) -> int | str | None:
    """Parse and validate num_samples: integer, 'auto', or percentage like '50%'.

    This is a shared utility used by both CLI argument parsing and config validation.

    Args:
        value: Raw value - can be int, string, or None

    Returns:
        Parsed value: int, "auto", percentage string like "50%", or None

    Raises:
        ValueError: If the value is invalid
    """
    if value is None:
        return None
    if isinstance(value, int):
        if value < 1:
            raise ValueError("num_samples must be at least 1")
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return "auto"
        if normalized.endswith("%"):
            try:
                pct = float(normalized[:-1])
            except ValueError as e:
                raise ValueError(f"Invalid percentage format: {value}") from e
            if pct <= 0:
                raise ValueError("Percentage must be greater than 0")
            return normalized
        # Try to parse as integer string
        try:
            parsed = int(normalized)
        except ValueError as e:
            raise ValueError(
                f"Invalid num_samples value: {value}. Use integer, 'auto', or percentage like '50%'"
            ) from e
        if parsed < 1:
            raise ValueError("num_samples must be at least 1")
        return parsed
    raise ValueError(f"num_samples must be int or string, got {type(value).__name__}")


def get_bool_env(key: str, default: bool = False) -> bool:
    """Get a boolean environment variable.

    Supports: '1', 'true', 'yes', 'on' (case-insensitive) as True.
    Everything else is False unless default is True and key is missing.
    """
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "on")


def import_optional_dependency(
    module_name: str,
    extra: str | None = None,
) -> Any:
    """
    Import an optional dependency at runtime.

    Args:
        module_name (str): The name of the module to import.
        extra (str | None): The optional dependency group providing this module.

    Returns:
        Any: The imported module.

    Raises:
        ModuleNotFoundError: If the module is not installed.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if extra:
            msg = (
                f"The '{module_name}' library is required for the '{extra}' features. "
                f"Please install it using: pip install 'deepfabric[{extra}]'"
            )
        else:
            msg = f"The '{module_name}' library is required but is not installed."
        raise ModuleNotFoundError(msg) from None
