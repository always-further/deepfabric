"""Native DeepFabric Dataset implementation.

This module provides a simple, maintainable Dataset class with no external
dependencies (beyond stdlib). It supports column-oriented access patterns
similar to HuggingFace datasets.
"""

import json
import random

from collections.abc import Callable, Iterator
from typing import Any, overload


class Dataset:
    """A simple, native dataset class that stores data as a list of dicts
    with column-oriented access patterns.

    Examples:
        >>> ds = Dataset([{"text": "hello"}, {"text": "world"}])
        >>> len(ds)
        2
        >>> ds["text"]
        ['hello', 'world']
        >>> ds[0]
        {'text': 'hello'}
        >>> ds[0:1]
        Dataset with 1 samples
    """

    def __init__(self, data: list[dict[str, Any]], metadata: dict | None = None):
        """Initialize dataset from list of sample dicts.

        Args:
            data: List of sample dictionaries
            metadata: Optional metadata (source, path, etc.)
        """
        self._data = data
        self._metadata = metadata or {}
        self._columns: list[str] | None = None

    @property
    def column_names(self) -> list[str]:
        """Return list of column names."""
        if self._columns is None:
            if self._data:
                # Collect all unique keys across samples
                all_keys: set[str] = set()
                for sample in self._data:
                    all_keys.update(sample.keys())
                self._columns = sorted(all_keys)
            else:
                self._columns = []
        return self._columns

    @property
    def num_rows(self) -> int:
        """Return number of samples (alias for len)."""
        return len(self._data)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self._data)

    @overload
    def __getitem__(self, key: str) -> list[Any]: ...

    @overload
    def __getitem__(self, key: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, key: slice) -> "Dataset": ...

    def __getitem__(self, key: str | int | slice) -> Any:
        """Access by column name, index, or slice.

        Args:
            key: Column name (str), row index (int), or slice

        Returns:
            - For str: list of values for that column
            - For int: dict for that sample
            - For slice: new Dataset with selected samples

        Examples:
            >>> ds["messages"]  # Get column as list
            >>> ds[0]           # Get first sample as dict
            >>> ds[0:10]        # Get first 10 samples as new Dataset
        """
        if isinstance(key, str):
            # Column access - return list of values
            return [sample.get(key) for sample in self._data]
        if isinstance(key, int):
            # Single sample access
            if key < 0:
                key = len(self._data) + key
            if key < 0 or key >= len(self._data):
                raise IndexError(
                    f"Index {key} out of range for dataset with {len(self._data)} samples"
                )
            return self._data[key]
        if isinstance(key, slice):
            # Slice access - return new Dataset
            return Dataset(self._data[key], self._metadata.copy())
        raise TypeError(f"Invalid key type: {type(key)}. Expected str, int, or slice.")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples."""
        return iter(self._data)

    def __repr__(self) -> str:
        """Return string representation."""
        cols = ", ".join(self.column_names[:5])
        if len(self.column_names) > 5:  # noqa: PLR2004
            cols += ", ..."
        return f"Dataset(num_rows={len(self)}, columns=[{cols}])"

    def split(
        self,
        test_size: float = 0.1,
        seed: int | None = None,
    ) -> dict[str, "Dataset"]:
        """Split dataset into train and test sets.

        Args:
            test_size: Fraction of data for test set (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Dict with "train" and "test" Dataset instances

        Examples:
            >>> splits = ds.split(test_size=0.1, seed=42)
            >>> train_ds = splits["train"]
            >>> test_ds = splits["test"]
        """
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be between 0.0 and 1.0 (exclusive)")

        # Use a local Random instance to avoid affecting global state
        rng = random.Random(seed)  # noqa: S311 # nosec

        # Create shuffled indices
        indices = list(range(len(self._data)))
        rng.shuffle(indices)

        # Calculate split point
        split_idx = int(len(indices) * (1 - test_size))

        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

        return {
            "train": self.select(train_indices),
            "test": self.select(test_indices),
        }

    def select(self, indices: list[int]) -> "Dataset":
        """Select samples by indices.

        Args:
            indices: List of integer indices to select

        Returns:
            New Dataset with selected samples
        """
        return Dataset([self._data[i] for i in indices], self._metadata.copy())

    def shuffle(self, seed: int | None = None) -> "Dataset":
        """Return a shuffled copy of the dataset.

        Args:
            seed: Random seed for reproducibility

        Returns:
            New Dataset with shuffled samples
        """
        rng = random.Random(seed)  #  nosec  # noqa: S311
        indices = list(range(len(self._data)))
        rng.shuffle(indices)
        return self.select(indices)

    def map(
        self,
        fn: Callable[[dict[str, Any]], dict[str, Any]],
        _num_proc: int | None = None,
        _desc: str | None = None,
        **_kwargs: Any,
    ) -> "Dataset":
        """Apply function to each sample.

        Args:
            fn: Function that takes a sample dict and returns a new sample dict
            _num_proc: Ignored (for HuggingFace Dataset compatibility)
            _desc: Ignored (for HuggingFace Dataset compatibility)
            **_kwargs: Additional kwargs ignored for compatibility

        Returns:
            New Dataset with transformed samples

        Examples:
            >>> ds.map(lambda x: {"text": x["text"].upper()})
        """
        return Dataset([fn(sample) for sample in self._data], self._metadata.copy())

    def filter(self, fn: Callable[[dict[str, Any]], bool]) -> "Dataset":
        """Filter samples by predicate function.

        Args:
            fn: Function that takes a sample dict and returns True to keep

        Returns:
            New Dataset with filtered samples

        Examples:
            >>> ds.filter(lambda x: len(x["text"]) > 10)
        """
        return Dataset([s for s in self._data if fn(s)], self._metadata.copy())

    def to_list(self) -> list[dict[str, Any]]:
        """Return data as list of dicts.

        Returns:
            Copy of internal data as list of dictionaries
        """
        return self._data.copy()

    def to_hf(self) -> Any:
        """Convert to HuggingFace Dataset for use with TRL/transformers.

        Returns:
            A HuggingFace datasets.Dataset instance

        Raises:
            ImportError: If the 'datasets' package is not installed

        Examples:
            >>> from deepfabric import load_dataset
            >>> ds = load_dataset("data.jsonl")
            >>> hf_ds = ds.to_hf()
            >>> trainer = SFTTrainer(train_dataset=hf_ds, ...)
        """
        try:
            from datasets import Dataset as HFDataset  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "The 'datasets' package is required for to_hf(). "
                "Install it with: pip install datasets"
            ) from None

        return HFDataset.from_list(self._data)

    def to_jsonl(self, path: str) -> None:
        """Save dataset to JSONL file.

        Args:
            path: File path to save to
        """
        with open(path, "w", encoding="utf-8") as f:
            for sample in self._data:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        """Load dataset from JSONL file.

        Args:
            path: File path to load from

        Returns:
            New Dataset loaded from file
        """
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return cls(data, metadata={"source": "jsonl", "path": path})

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> "Dataset":
        """Create dataset from list of dicts.

        Args:
            data: List of sample dictionaries

        Returns:
            New Dataset from the provided data
        """
        return cls(data)


class DatasetDict(dict):
    """Dictionary of Dataset objects for train/test/validation splits.

    A simple dict subclass that provides typed access to Dataset values.

    Examples:
        >>> dd = DatasetDict({"train": train_ds, "test": test_ds})
        >>> dd["train"]
        Dataset(num_rows=100, columns=[...])
    """

    def __getitem__(self, key: str) -> Dataset:
        """Get Dataset by split name."""
        return super().__getitem__(key)

    def __repr__(self) -> str:
        """Return string representation."""
        splits = ", ".join(f"{k}: {len(v)} rows" for k, v in self.items())
        return f"DatasetDict({{{splits}}})"
