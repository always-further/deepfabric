"""Dataset loading functionality for DeepFabric.

This module provides the load_dataset function for loading datasets from:
- Local text files (with line/paragraph/document sampling)
- Local JSONL files
- DeepFabric Cloud (via namespace/slug format)
"""

import json
import re
import warnings

from http import HTTPStatus
from pathlib import Path
from typing import Any, Literal

import httpx

from .auth import DEFAULT_API_URL, get_stored_token
from .dataset import Dataset, DatasetDict
from .exceptions import LoaderError

# Default cache directory for cloud datasets
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "deepfabric" / "datasets"


def _detect_source(path: str) -> Literal["local", "cloud"]:
    """Detect if path refers to local files or DeepFabric Cloud.

    Args:
        path: The path argument to load_dataset

    Returns:
        "local" for local file loading, "cloud" for cloud loading
    """
    # Known local format types
    if path.lower() in ("text", "json", "jsonl", "csv"):
        return "local"

    # Cloud pattern: namespace/slug (alphanumeric with hyphens/underscores, single slash)
    cloud_pattern = re.compile(r"^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$")
    if cloud_pattern.match(path) and not Path(path).exists():
        return "cloud"

    # Default to local (file path)
    return "local"


def _read_text_file(file_path: str, sample_by: str) -> list[str]:
    """Read text file with specified sampling strategy.

    Args:
        file_path: Path to the text file
        sample_by: Sampling strategy - "line", "paragraph", or "document"

    Returns:
        List of text samples
    """
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    if sample_by == "document":
        # Entire file as one sample
        return [content.strip()] if content.strip() else []
    if sample_by == "paragraph":
        # Split on double newlines (paragraph boundaries)
        paragraphs = re.split(r"\n\s*\n", content)
        return [p.strip() for p in paragraphs if p.strip()]
    # Default: line-by-line
    return [line.strip() for line in content.split("\n") if line.strip()]


def _normalize_data_files(
    data_files: dict[str, str | list[str]] | str | list[str] | None,
    data_dir: str | None,
) -> dict[str, list[str]]:
    """Normalize data_files to a consistent dict format.

    Args:
        data_files: Input data_files in various formats
        data_dir: Optional directory prefix

    Returns:
        Dict mapping split names to lists of file paths
    """
    if data_files is None:
        return {}

    # Normalize to dict format
    if isinstance(data_files, str):
        files_dict: dict[str, list[str]] = {"train": [data_files]}
    elif isinstance(data_files, list):
        files_dict = {"train": data_files}
    else:
        files_dict = {k: [v] if isinstance(v, str) else list(v) for k, v in data_files.items()}

    # Apply data_dir prefix if provided
    if data_dir:
        files_dict = {k: [str(Path(data_dir) / f) for f in v] for k, v in files_dict.items()}

    return files_dict


def _load_text_files(
    data_files: dict[str, str | list[str]] | str | list[str] | None,
    data_dir: str | None,
    sample_by: str,
) -> Dataset | DatasetDict:
    """Load text files into Dataset.

    Args:
        data_files: File paths specification
        data_dir: Optional directory prefix
        sample_by: Sampling strategy

    Returns:
        Dataset or DatasetDict
    """
    files_dict = _normalize_data_files(data_files, data_dir)

    if not files_dict:
        raise LoaderError("No data files specified for text format")

    # Load each split
    datasets: dict[str, Dataset] = {}
    for split_name, file_list in files_dict.items():
        samples: list[dict[str, Any]] = []
        for file_path in file_list:
            if not Path(file_path).exists():
                raise LoaderError(f"File not found: {file_path}")
            texts = _read_text_file(file_path, sample_by)
            samples.extend([{"text": t} for t in texts])
        datasets[split_name] = Dataset(samples, {"source": "text", "split": split_name})

    # Return single Dataset if only train split
    if len(datasets) == 1 and "train" in datasets:
        return datasets["train"]
    return DatasetDict(datasets)


def _load_json_file(file_path: str) -> list[dict[str, Any]]:
    """Load a standard JSON file (array of objects).

    Args:
        file_path: Path to the JSON file

    Returns:
        List of sample dictionaries

    Raises:
        LoaderError: If the file is not valid JSON or not an array of objects
    """
    with open(file_path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise LoaderError(
                f"Invalid JSON in {file_path}: {e}",
                context={"file": file_path},
            ) from e

    if isinstance(data, list):
        # Validate all items are dicts
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise LoaderError(
                    f"Expected array of objects in {file_path}, "
                    f"but item at index {i} is {type(item).__name__}",
                    context={"file": file_path, "index": i},
                )
        return data
    if isinstance(data, dict):
        # Single object - wrap in list
        return [data]
    raise LoaderError(
        f"Expected JSON array or object in {file_path}, got {type(data).__name__}",
        context={"file": file_path},
    )


def _load_jsonl_file(file_path: str) -> list[dict[str, Any]]:
    """Load a JSONL file (one JSON object per line).

    Args:
        file_path: Path to the JSONL file

    Returns:
        List of sample dictionaries
    """
    samples: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LoaderError(
                    f"Invalid JSON on line {line_num} of {file_path}: {e}",
                    context={"file": file_path, "line": line_num},
                ) from e
    return samples


def _load_json_or_jsonl_file(file_path: str) -> list[dict[str, Any]]:
    """Load a JSON or JSONL file based on extension.

    Args:
        file_path: Path to the file

    Returns:
        List of sample dictionaries
    """
    path = Path(file_path)
    if path.suffix == ".jsonl":
        return _load_jsonl_file(file_path)
    # .json files are standard JSON (array or object)
    return _load_json_file(file_path)


def _load_jsonl_files(
    data_files: dict[str, str | list[str]] | str | list[str] | None,
    data_dir: str | None,
) -> Dataset | DatasetDict:
    """Load JSON/JSONL files into Dataset.

    Args:
        data_files: File paths specification
        data_dir: Optional directory prefix

    Returns:
        Dataset or DatasetDict
    """
    files_dict = _normalize_data_files(data_files, data_dir)

    if not files_dict:
        raise LoaderError("No data files specified for json/jsonl format")

    datasets: dict[str, Dataset] = {}
    for split_name, file_list in files_dict.items():
        samples: list[dict[str, Any]] = []
        for file_path in file_list:
            if not Path(file_path).exists():
                raise LoaderError(f"File not found: {file_path}")
            samples.extend(_load_json_or_jsonl_file(file_path))
        datasets[split_name] = Dataset(samples, {"source": "json", "split": split_name})

    if len(datasets) == 1 and "train" in datasets:
        return datasets["train"]
    return DatasetDict(datasets)


def _load_from_directory(
    data_dir: str,
    sample_by: str,
) -> Dataset | DatasetDict:
    """Load all files from a directory.

    Args:
        data_dir: Directory path
        sample_by: Sampling strategy for text files

    Returns:
        Dataset or DatasetDict
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise LoaderError(f"Directory not found: {data_dir}")

    # Find all supported files
    text_files = list(dir_path.glob("*.txt"))
    jsonl_files = list(dir_path.glob("*.jsonl"))
    json_files = list(dir_path.glob("*.json"))

    if not text_files and not jsonl_files and not json_files:
        raise LoaderError(f"No .txt, .json, or .jsonl files found in {data_dir}")

    samples: list[dict[str, Any]] = []

    # Load text files
    for file_path in text_files:
        texts = _read_text_file(str(file_path), sample_by)
        samples.extend([{"text": t} for t in texts])

    # Load JSONL files (one JSON object per line)
    for file_path in jsonl_files:
        samples.extend(_load_jsonl_file(str(file_path)))

    # Load JSON files (array of objects or single object)
    for file_path in json_files:
        samples.extend(_load_json_file(str(file_path)))

    return Dataset(samples, {"source": "directory", "path": data_dir})


def _get_cache_path(namespace: str, slug: str, cache_dir: Path) -> Path:
    """Get cache file path for a cloud dataset.

    Args:
        namespace: Dataset namespace
        slug: Dataset slug
        cache_dir: Base cache directory

    Returns:
        Path to cached JSONL file
    """
    return cache_dir / f"{namespace}_{slug}.jsonl"


def _load_from_cloud(
    path: str,
    split: str | None,
    token: str | None,
    api_url: str | None,
    use_cache: bool,
    streaming: bool,
) -> Dataset:
    """Load dataset from DeepFabric Cloud.

    Args:
        path: Dataset path in "namespace/slug" format
        split: Optional split name (not used yet, reserved for future)
        token: Optional auth token (uses stored token if not provided)
        api_url: Optional API URL (uses default if not provided)
        use_cache: Whether to use/store cached data
        streaming: Whether to stream the dataset (not yet implemented on client side)

    Returns:
        Dataset loaded from cloud
    """
    # TODO: Implement streaming using Parquet shards endpoint
    # Backend supports: GET /api/v1/datasets/{id}/parquet (manifest)
    # and GET /api/v1/datasets/{id}/parquet/{filename} (shard with Range support)
    if streaming:
        warnings.warn(
            "streaming=True is not yet implemented. "
            "Falling back to loading entire dataset into memory. "
            "For large datasets, this may cause memory issues.",
            UserWarning,
            stacklevel=3,
        )

    # TODO: Implement server-side split support
    # For now, use dataset.split() after loading
    if split:
        warnings.warn(
            f"split='{split}' is not yet implemented for cloud datasets. "
            "Use dataset.split() after loading instead.",
            UserWarning,
            stacklevel=3,
        )

    # Parse namespace/slug
    parts = path.split("/")
    if len(parts) != 2:  # noqa: PLR2004
        raise LoaderError(
            f"Invalid cloud path format: {path}. Expected 'namespace/slug'.",
            context={"path": path},
        )
    namespace, slug = parts

    effective_token = token or get_stored_token()
    effective_api_url = api_url or DEFAULT_API_URL

    # Check cache first if enabled
    cache_path = _get_cache_path(namespace, slug, DEFAULT_CACHE_DIR)
    if use_cache and cache_path.exists():
        return Dataset.from_jsonl(str(cache_path))

    # Build request headers
    headers: dict[str, str] = {}
    if effective_token:
        headers["Authorization"] = f"Bearer {effective_token}"

    # Fetch from API
    try:
        with httpx.Client() as client:
            response = client.get(
                f"{effective_api_url}/api/v1/datasets/by-slug/{namespace}/{slug}/with-samples",
                headers=headers,
                timeout=120.0,
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == HTTPStatus.NOT_FOUND:
            raise LoaderError(
                f"Dataset not found: {path}",
                context={"namespace": namespace, "slug": slug},
            ) from e
        if e.response.status_code == HTTPStatus.UNAUTHORIZED:
            raise LoaderError(
                f"Authentication required for dataset: {path}. "
                "Run 'deepfabric auth login' or pass token parameter.",
                context={"namespace": namespace, "slug": slug},
            ) from e
        if e.response.status_code == HTTPStatus.FORBIDDEN:
            raise LoaderError(
                f"Access denied for dataset: {path}. "
                "You may not have permission to access this private dataset.",
                context={"namespace": namespace, "slug": slug},
            ) from e
        raise LoaderError(
            f"Failed to load dataset from cloud: {e}",
            context={"path": path, "status_code": e.response.status_code},
        ) from e
    except httpx.RequestError as e:
        raise LoaderError(
            f"Network error while loading dataset: {e}",
            context={"path": path},
        ) from e

    # Extract samples from response
    samples = data.get("samples", [])
    if not samples:
        raise LoaderError(
            f"Dataset is empty: {path}",
            context={"namespace": namespace, "slug": slug},
        )

    # Extract sample data - API may wrap in {"data": ...} format
    sample_data: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, dict) and "data" in sample:
            sample_data.append(sample["data"])
        else:
            sample_data.append(sample)

    dataset = Dataset(sample_data, {"source": "cloud", "path": path})

    # Cache the dataset by default
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_jsonl(str(cache_path))

    return dataset


def load_dataset(
    path: str,
    *,
    data_files: dict[str, str | list[str]] | str | list[str] | None = None,
    data_dir: str | None = None,
    split: str | None = None,
    sample_by: Literal["line", "paragraph", "document"] = "line",
    use_cache: bool = True,
    token: str | None = None,
    api_url: str | None = None,
    streaming: bool = False,
) -> Dataset | DatasetDict:
    """Load a dataset from local files or DeepFabric Cloud.

    Args:
        path: Dataset path. Can be:
            - "text" for local text files (requires data_files or data_dir)
            - "json" or "jsonl" for local JSON/JSONL files
            - "namespace/slug" for DeepFabric Cloud datasets
            - Direct file path (e.g., "data.jsonl")
        data_files: File paths for local loading. Can be:
            - Single path: "train.txt"
            - List of paths: ["file1.txt", "file2.txt"]
            - Dict for splits: {"train": "train.txt", "test": "test.txt"}
        data_dir: Directory containing data files, or directory to load from
        split: Which split to load ("train", "test", "validation")
        sample_by: How to sample text files:
            - "line": Each line is a sample (default)
            - "paragraph": Split on double newlines
            - "document": Entire file is one sample
        use_cache: Cache cloud datasets locally (default True).
            Cache location: ~/.cache/deepfabric/datasets/
        token: DeepFabric Cloud auth token (defaults to stored token)
        api_url: DeepFabric API URL (defaults to production)
        streaming: If True, return an iterable dataset (cloud only, not yet implemented)

    Returns:
        Dataset or DatasetDict depending on input structure

    Raises:
        LoaderError: If loading fails (file not found, invalid format, auth failure, etc.)

    Examples:
        >>> from deepfabric import load_dataset
        >>>
        >>> # Load from local text files
        >>> ds = load_dataset("text", data_files={"train": "train.txt", "test": "test.txt"})
        >>>
        >>> # Load with paragraph sampling
        >>> ds = load_dataset("text", data_files="my_text.txt", sample_by="paragraph")
        >>>
        >>> # Load from DeepFabric Cloud
        >>> ds = load_dataset("username/my-dataset")
        >>>
        >>> # Access data
        >>> messages = ds["messages"]
        >>> first_sample = ds[0]
        >>>
        >>> # Split into train/test
        >>> splits = ds.split(test_size=0.1, seed=42)
    """
    source = _detect_source(path)

    if source == "cloud":
        return _load_from_cloud(path, split, token, api_url, use_cache, streaming)

    # Local loading
    if path.lower() == "text":
        if not data_files and not data_dir:
            raise LoaderError(
                "text format requires data_files or data_dir parameter",
                context={"path": path},
            )
        if data_dir and not data_files:
            dataset = _load_from_directory(data_dir, sample_by)
        else:
            dataset = _load_text_files(data_files, data_dir, sample_by)

    elif path.lower() in ("json", "jsonl"):
        if not data_files and not data_dir:
            raise LoaderError(
                f"{path} format requires data_files or data_dir parameter",
                context={"path": path},
            )
        if data_dir and not data_files:
            dataset = _load_from_directory(data_dir, sample_by)
        else:
            dataset = _load_jsonl_files(data_files, data_dir)

    else:
        # Assume it's a direct file path
        file_path = Path(path)
        if file_path.is_file():
            if file_path.suffix in (".jsonl", ".json"):
                dataset = _load_jsonl_files(str(file_path), None)
            else:
                dataset = _load_text_files(str(file_path), None, sample_by)
        elif file_path.is_dir():
            dataset = _load_from_directory(str(file_path), sample_by)
        else:
            raise LoaderError(
                f"Path not found: {path}",
                context={"path": path},
            )

    # Handle split parameter for DatasetDict
    if split is not None and isinstance(dataset, DatasetDict):
        if split not in dataset:
            available = list(dataset.keys())
            raise LoaderError(
                f"Split '{split}' not found. Available splits: {available}",
                context={"path": path, "split": split, "available": available},
            )
        return dataset[split]

    return dataset
