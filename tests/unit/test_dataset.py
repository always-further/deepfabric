"""Tests for the native Dataset class."""

import json
import tempfile

from pathlib import Path

import pytest

from deepfabric.dataset import Dataset, DatasetDict


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return [
        {"text": "hello", "label": 1},
        {"text": "world", "label": 0},
        {"text": "test", "label": 1},
    ]


@pytest.fixture
def sample_dataset(sample_data):
    """Sample dataset for testing."""
    return Dataset(sample_data)


@pytest.fixture
def messages_data():
    """Sample messages data for chat format testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thanks!"},
            ]
        },
    ]


class TestDatasetBasics:
    """Test basic Dataset functionality."""

    def test_dataset_len(self, sample_dataset):
        """Test __len__ returns correct count."""
        assert len(sample_dataset) == 3  # noqa: PLR2004

    def test_dataset_num_rows(self, sample_dataset):
        """Test num_rows property."""
        assert sample_dataset.num_rows == 3  # noqa: PLR2004

    def test_dataset_empty(self):
        """Test empty dataset."""
        ds = Dataset([])
        assert len(ds) == 0
        assert ds.column_names == []

    def test_dataset_column_names(self, sample_dataset):
        """Test column_names property."""
        assert sorted(sample_dataset.column_names) == ["label", "text"]

    def test_dataset_column_names_heterogeneous(self):
        """Test column_names with heterogeneous data."""
        data = [
            {"a": 1},
            {"a": 2, "b": 3},
            {"c": 4},
        ]
        ds = Dataset(data)
        assert sorted(ds.column_names) == ["a", "b", "c"]


class TestDatasetAccess:
    """Test Dataset access patterns."""

    def test_column_access(self, sample_dataset):
        """Test accessing a column by name."""
        texts = sample_dataset["text"]
        assert texts == ["hello", "world", "test"]

    def test_column_access_missing_key(self, sample_dataset):
        """Test accessing missing column returns None values."""
        missing = sample_dataset["nonexistent"]
        assert missing == [None, None, None]

    def test_index_access(self, sample_dataset):
        """Test accessing by index."""
        first = sample_dataset[0]
        assert first == {"text": "hello", "label": 1}

    def test_index_access_negative(self, sample_dataset):
        """Test negative index access."""
        last = sample_dataset[-1]
        assert last == {"text": "test", "label": 1}

    def test_index_access_out_of_range(self, sample_dataset):
        """Test out of range index raises error."""
        with pytest.raises(IndexError):
            _ = sample_dataset[100]

    def test_slice_access(self, sample_dataset):
        """Test slice access returns new Dataset."""
        sliced = sample_dataset[0:2]
        assert isinstance(sliced, Dataset)
        assert len(sliced) == 2  # noqa: PLR2004
        assert sliced["text"] == ["hello", "world"]

    def test_slice_access_step(self, sample_dataset):
        """Test slice with step."""
        sliced = sample_dataset[::2]
        assert len(sliced) == 2  # noqa: PLR2004
        assert sliced["text"] == ["hello", "test"]

    def test_invalid_key_type(self, sample_dataset):
        """Test invalid key type raises error."""
        with pytest.raises(TypeError):
            _ = sample_dataset[1.5]


class TestDatasetIteration:
    """Test Dataset iteration."""

    def test_iteration(self, sample_dataset):
        """Test iterating over dataset."""
        items = list(sample_dataset)
        assert len(items) == 3  # noqa: PLR2004
        assert items[0] == {"text": "hello", "label": 1}

    def test_iteration_in_for_loop(self, sample_dataset):
        """Test iteration in for loop."""
        texts = []
        for sample in sample_dataset:
            texts.append(sample["text"])
        assert texts == ["hello", "world", "test"]


class TestDatasetSplit:
    """Test Dataset split functionality."""

    def test_split_basic(self):
        """Test basic split."""
        # Use larger dataset for meaningful split
        data = [{"i": i} for i in range(100)]
        ds = Dataset(data)
        splits = ds.split(test_size=0.2, seed=42)

        assert "train" in splits
        assert "test" in splits
        assert len(splits["train"]) == 80  # noqa: PLR2004
        assert len(splits["test"]) == 20  # noqa: PLR2004

    def test_split_seed_reproducibility(self):
        """Test split with same seed produces same results."""
        data = [{"i": i} for i in range(100)]
        ds = Dataset(data)

        splits1 = ds.split(test_size=0.2, seed=42)
        splits2 = ds.split(test_size=0.2, seed=42)

        assert splits1["train"].to_list() == splits2["train"].to_list()
        assert splits1["test"].to_list() == splits2["test"].to_list()

    def test_split_different_seeds(self):
        """Test split with different seeds produces different results."""
        data = [{"i": i} for i in range(100)]
        ds = Dataset(data)

        splits1 = ds.split(test_size=0.2, seed=42)
        splits2 = ds.split(test_size=0.2, seed=123)

        # Different seeds should produce different orderings
        assert splits1["train"].to_list() != splits2["train"].to_list()

    def test_split_invalid_test_size_zero(self):
        """Test split with invalid test_size raises error."""
        ds = Dataset([{"i": i} for i in range(10)])
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.split(test_size=0.0)

    def test_split_invalid_test_size_one(self):
        """Test split with test_size=1.0 raises error."""
        ds = Dataset([{"i": i} for i in range(10)])
        with pytest.raises(ValueError, match="test_size must be between"):
            ds.split(test_size=1.0)


class TestDatasetTransformations:
    """Test Dataset transformation methods."""

    def test_select(self, sample_dataset):
        """Test select by indices."""
        selected = sample_dataset.select([0, 2])
        assert len(selected) == 2  # noqa: PLR2004
        assert selected["text"] == ["hello", "test"]

    def test_shuffle(self):
        """Test shuffle with seed."""
        data = [{"i": i} for i in range(10)]
        ds = Dataset(data)

        shuffled = ds.shuffle(seed=42)
        assert len(shuffled) == 10  # noqa: PLR2004
        # Should be in different order
        assert shuffled.to_list() != ds.to_list()

    def test_shuffle_reproducibility(self):
        """Test shuffle with same seed is reproducible."""
        data = [{"i": i} for i in range(10)]
        ds = Dataset(data)

        shuffled1 = ds.shuffle(seed=42)
        shuffled2 = ds.shuffle(seed=42)
        assert shuffled1.to_list() == shuffled2.to_list()

    def test_map(self, sample_dataset):
        """Test map function."""
        mapped = sample_dataset.map(lambda x: {"text": x["text"].upper(), "label": x["label"]})
        assert mapped["text"] == ["HELLO", "WORLD", "TEST"]

    def test_filter(self, sample_dataset):
        """Test filter function."""
        filtered = sample_dataset.filter(lambda x: x["label"] == 1)
        assert len(filtered) == 2  # noqa: PLR2004
        assert filtered["text"] == ["hello", "test"]


class TestDatasetSerialization:
    """Test Dataset serialization."""

    def test_to_list(self, sample_data, sample_dataset):
        """Test to_list returns copy."""
        result = sample_dataset.to_list()
        assert result == sample_data
        # Should be a copy
        result.append({"new": "item"})
        assert len(sample_dataset) == 3  # noqa: PLR2004

    def test_to_jsonl(self, sample_dataset):
        """Test saving to JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            sample_dataset.to_jsonl(path)

            # Read back and verify
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # noqa: PLR2004
            assert json.loads(lines[0]) == {"text": "hello", "label": 1}
        finally:
            Path(path).unlink()

    def test_from_jsonl(self, sample_data):
        """Test loading from JSONL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
            path = f.name

        try:
            ds = Dataset.from_jsonl(path)
            assert len(ds) == 3  # noqa: PLR2004
            assert ds["text"] == ["hello", "world", "test"]
        finally:
            Path(path).unlink()

    def test_from_list(self, sample_data):
        """Test creating from list."""
        ds = Dataset.from_list(sample_data)
        assert len(ds) == 3  # noqa: PLR2004
        assert ds["text"] == ["hello", "world", "test"]


class TestDatasetRepr:
    """Test Dataset string representation."""

    def test_repr(self, sample_dataset):
        """Test __repr__ format."""
        repr_str = repr(sample_dataset)
        assert "Dataset" in repr_str
        assert "num_rows=3" in repr_str

    def test_repr_many_columns(self):
        """Test __repr__ with many columns truncates."""
        data = [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}]
        ds = Dataset(data)
        repr_str = repr(ds)
        assert "..." in repr_str


class TestDatasetDict:
    """Test DatasetDict functionality."""

    def test_datasetdict_creation(self, sample_data):
        """Test creating DatasetDict."""
        train_ds = Dataset(sample_data[:2])
        test_ds = Dataset(sample_data[2:])
        dd = DatasetDict({"train": train_ds, "test": test_ds})

        assert len(dd["train"]) == 2  # noqa: PLR2004
        assert len(dd["test"]) == 1

    def test_datasetdict_repr(self, sample_data):
        """Test DatasetDict __repr__."""
        train_ds = Dataset(sample_data[:2])
        test_ds = Dataset(sample_data[2:])
        dd = DatasetDict({"train": train_ds, "test": test_ds})

        repr_str = repr(dd)
        assert "DatasetDict" in repr_str
        assert "train" in repr_str
        assert "test" in repr_str


class TestMessagesFormat:
    """Test Dataset with chat messages format."""

    def test_messages_column_access(self, messages_data):
        """Test accessing messages column."""
        ds = Dataset(messages_data)
        messages = ds["messages"]
        assert len(messages) == 2  # noqa: PLR2004
        assert messages[0][0]["role"] == "user"

    def test_messages_iteration_for_tokenizer(self, messages_data):
        """Test iteration pattern for use with tokenizer."""
        ds = Dataset(messages_data)

        # Simulate tokenizer.apply_chat_template pattern
        results = []
        for msg_list in ds["messages"]:
            # Just check we can iterate over messages
            result = " ".join(m["content"] for m in msg_list)
            results.append(result)

        assert results[0] == "Hello Hi there!"
        assert results[1] == "How are you? I'm doing well, thanks!"
