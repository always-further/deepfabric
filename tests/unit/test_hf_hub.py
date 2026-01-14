"""Tests for the HF Hub uploader module."""

from unittest.mock import Mock, patch

import pytest

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from requests import Request, Response

from deepfabric.hf_hub import HFUploader


@pytest.fixture
def mock_dataset_card():
    """Create a mock dataset card."""
    card = Mock()
    card.data.tags = []
    return card


@pytest.fixture
def uploader():
    """Create an HFUploader instance."""
    return HFUploader("dummy_token")


def test_update_dataset_card(uploader, mock_dataset_card):
    """Test updating dataset card with tags."""
    with patch("deepfabric.hf_hub.DatasetCard") as mock_card_class:
        mock_card_class.load.return_value = mock_dataset_card

        # Test with default tags only
        uploader.update_dataset_card("test/repo")
        assert "deepfabric" in mock_dataset_card.data.tags
        assert "synthetic" in mock_dataset_card.data.tags
        mock_dataset_card.push_to_hub.assert_called_once_with("test/repo", token="dummy_token")  # noqa: S106

        # Reset mock
        mock_dataset_card.data.tags = []
        mock_dataset_card.push_to_hub.reset_mock()

        # Test with custom tags
        custom_tags = ["custom1", "custom2"]
        uploader.update_dataset_card("test/repo", tags=custom_tags)
        assert all(
            tag in mock_dataset_card.data.tags for tag in ["deepfabric", "synthetic"] + custom_tags
        )
        mock_dataset_card.push_to_hub.assert_called_once_with("test/repo", token="dummy_token")  # noqa: S106


def test_update_dataset_card_no_duplicate_tags(uploader, mock_dataset_card):
    """Test that 'deepfabric' tag is never duplicated regardless of source."""
    with patch("deepfabric.hf_hub.DatasetCard") as mock_card_class:
        mock_card_class.load.return_value = mock_dataset_card

        # Test 1: User explicitly includes "deepfabric" in custom tags
        mock_dataset_card.data.tags = []
        uploader.update_dataset_card("test/repo", tags=["deepfabric", "custom"])
        assert mock_dataset_card.data.tags.count("deepfabric") == 1
        assert "custom" in mock_dataset_card.data.tags
        assert "synthetic" in mock_dataset_card.data.tags

        # Test 2: Dataset card already has "deepfabric" from previous push
        mock_dataset_card.data.tags = ["deepfabric", "existing"]
        uploader.update_dataset_card("test/repo", tags=["new_tag"])
        assert mock_dataset_card.data.tags.count("deepfabric") == 1
        assert "existing" in mock_dataset_card.data.tags
        assert "new_tag" in mock_dataset_card.data.tags
        assert "synthetic" in mock_dataset_card.data.tags

        # Test 3: User includes "deepfabric" AND card already has it
        mock_dataset_card.data.tags = ["deepfabric"]
        uploader.update_dataset_card("test/repo", tags=["deepfabric", "another"])
        assert mock_dataset_card.data.tags.count("deepfabric") == 1
        assert "another" in mock_dataset_card.data.tags
        assert "synthetic" in mock_dataset_card.data.tags

        # Test 4: Multiple duplicate tags in custom tags
        mock_dataset_card.data.tags = []
        uploader.update_dataset_card(
            "test/repo", tags=["deepfabric", "custom", "deepfabric", "synthetic"]
        )
        assert mock_dataset_card.data.tags.count("deepfabric") == 1
        assert mock_dataset_card.data.tags.count("synthetic") == 1
        assert mock_dataset_card.data.tags.count("custom") == 1


def test_push_to_hub_success(uploader):
    """Test successful dataset push to hub."""
    with (
        patch("deepfabric.hf_hub.login") as mock_login,
        patch("deepfabric.hf_hub.HfApi") as mock_hf_api_class,
        patch.object(uploader, "update_dataset_card") as mock_update_card,
        patch.object(
            uploader, "_clean_dataset_for_upload", return_value="test.jsonl"
        ) as mock_clean,
    ):
        mock_api = Mock()
        mock_hf_api_class.return_value = mock_api

        result = uploader.push_to_hub("test/repo", "test.jsonl", tags=["test"])

        mock_login.assert_called_once_with(token="dummy_token")  # noqa: S106
        mock_clean.assert_called_once_with("test.jsonl")
        mock_api.create_repo.assert_called_once_with(
            repo_id="test/repo",
            repo_type="dataset",
            exist_ok=True,
            token="dummy_token",  # noqa: S106
        )
        mock_api.upload_file.assert_called_once_with(
            path_or_fileobj="test.jsonl",
            path_in_repo="data/train.jsonl",
            repo_id="test/repo",
            repo_type="dataset",
            token="dummy_token",  # noqa: S106
        )
        mock_update_card.assert_called_once()

        assert result["status"] == "success"
        assert "test/repo" in result["message"]


def test_push_to_hub_file_not_found(uploader):
    """Test push to hub with non-existent file."""
    with (
        patch("deepfabric.hf_hub.login") as _mock_login,
        patch("deepfabric.hf_hub.HfApi") as mock_hf_api_class,
    ):
        mock_api = Mock()
        mock_hf_api_class.return_value = mock_api
        mock_api.upload_file.side_effect = FileNotFoundError("File not found")

        result = uploader.push_to_hub("test/repo", "nonexistent.jsonl")
        assert result["status"] == "error"
        assert "not found" in result["message"]


@patch("deepfabric.hf_hub.login")
def test_push_to_hub_repository_not_found(mock_login, uploader):
    """Test push to hub with non-existent repository."""
    # Create a mock response object with all required attributes
    mock_response = Mock(spec=Response)
    mock_response.headers = {"x-request-id": "test-id"}
    mock_response.request = Mock(spec=Request)
    mock_response.status_code = 404
    mock_response.text = "Repository not found"

    mock_login.side_effect = RepositoryNotFoundError("Repository not found", response=mock_response)

    result = uploader.push_to_hub("nonexistent/repo", "test.jsonl")
    assert result["status"] == "error"
    assert "Repository" in result["message"]


@patch("deepfabric.hf_hub.login")
def test_push_to_hub_http_error(mock_login, uploader):
    """Test push to hub with HTTP error."""
    # Create a mock response object with all required attributes
    mock_response = Mock(spec=Response)
    mock_response.headers = {"x-request-id": "test-id"}
    mock_response.request = Mock(spec=Request)
    mock_response.status_code = 400
    mock_response.text = "Error message"

    # Create HfHubHTTPError with mock response
    mock_login.side_effect = HfHubHTTPError("HTTP Error", response=mock_response)

    result = uploader.push_to_hub("test/repo", "test.jsonl")
    assert result["status"] == "error"
    assert "HTTP Error" in result["message"]
