"""Tests for the cloud upload module."""

import os

from unittest.mock import MagicMock, Mock, patch

import pytest

from deepfabric.cloud_upload import (
    build_urls,
    derive_name_and_slug,
    get_current_user,
    handle_cloud_upload,
    prompt_for_name,
    upload_dataset,
    upload_topic_graph,
)


class TestDeriveNameAndSlug:
    """Tests for derive_name_and_slug function."""

    def test_simple_filename(self):
        """Test deriving name and slug from a simple filename."""
        name, slug = derive_name_and_slug("my-dataset.jsonl")
        assert name == "My Dataset"
        assert slug == "my-dataset"

    def test_underscore_filename(self):
        """Test deriving name and slug from a filename with underscores."""
        name, slug = derive_name_and_slug("my_custom_dataset.jsonl")
        assert name == "My Custom Dataset"
        assert slug == "my-custom-dataset"

    def test_complex_filename(self):
        """Test deriving name and slug from a complex filename."""
        name, slug = derive_name_and_slug("My Dataset v2.0 (final).jsonl")
        assert "My" in name
        assert slug == "my-dataset-v2-0-final"

    def test_json_extension(self):
        """Test handling of .json extension."""
        name, slug = derive_name_and_slug("topic_graph.json")
        assert name == "Topic Graph"
        assert slug == "topic-graph"

    def test_empty_after_cleaning(self):
        """Test handling when filename becomes empty after cleaning."""
        name, slug = derive_name_and_slug("....jsonl")
        assert slug == "unnamed-dataset"
        assert name == "Unnamed Dataset"

    def test_path_with_directory(self):
        """Test handling of path with directories."""
        name, slug = derive_name_and_slug("/path/to/my-dataset.jsonl")
        assert name == "My Dataset"
        assert slug == "my-dataset"


class TestBuildUrls:
    """Tests for build_urls function."""

    def test_dataset_with_username(self):
        """Test building URLs for dataset with username."""
        public_url, internal_url = build_urls(
            resource_type="datasets",
            resource_id="abc123",
            slug="my-dataset",
            username="testuser",
            frontend_url="https://example.deepfabric.null",
        )
        assert public_url == "https://example.deepfabric.null/dataset/testuser/my-dataset"
        assert internal_url == "https://example.deepfabric.null/datasets/abc123"

    def test_dataset_without_username(self):
        """Test building URLs for dataset without username."""
        public_url, internal_url = build_urls(
            resource_type="datasets",
            resource_id="abc123",
            slug="my-dataset",
            username=None,
            frontend_url="https://example.deepfabric.null",
        )
        assert public_url is None
        assert internal_url == "https://example.deepfabric.null/datasets/abc123"

    def test_graph_with_username(self):
        """Test building URLs for graph with username."""
        public_url, internal_url = build_urls(
            resource_type="graphs",
            resource_id="def456",
            slug="my-graph",
            username="testuser",
            frontend_url="https://example.deepfabric.null",
        )
        assert public_url == "https://example.deepfabric.null/graphs/testuser/my-graph"
        assert internal_url == "https://example.deepfabric.null/graphs/def456"


class TestGetCurrentUser:
    """Tests for get_current_user function."""

    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_no_token(self, mock_get_token):
        """Test when no auth token is available."""
        mock_get_token.return_value = None
        result = get_current_user()
        assert result is None

    @patch("deepfabric.cloud_upload.get_auth_token")
    @patch("deepfabric.cloud_upload.get_config")
    def test_cached_user(self, mock_get_config, mock_get_token):
        """Test when user info is cached."""
        mock_get_token.return_value = "test-token"
        mock_get_config.return_value = {
            "username": "cacheduser",
            "user_id": "cached-id",
            "email": "cached@test.com",
            "name": "Cached User",
        }
        result = get_current_user()
        assert result["username"] == "cacheduser"  # type: ignore
        assert result["id"] == "cached-id"  # type: ignore

    @patch("deepfabric.cloud_upload.save_config")
    @patch("deepfabric.cloud_upload.get_config")
    @patch("deepfabric.cloud_upload.get_auth_token")
    @patch("httpx.Client")
    def test_fetch_from_api(
        self, mock_client_class, mock_get_token, mock_get_config, mock_save_config
    ):
        """Test fetching user info from API."""
        mock_get_token.return_value = "test-token"
        mock_get_config.return_value = {}

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "api-id",
            "username": "apiuser",
            "email": "api@test.com",
            "name": "API User",
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = get_current_user()
        assert result["username"] == "apiuser"  # type: ignore
        assert result["id"] == "api-id"  # type: ignore
        mock_save_config.assert_called_once()


class TestUploadDataset:
    """Tests for upload_dataset function."""

    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_not_authenticated(self, mock_get_token, tmp_path):
        """Test upload when not authenticated."""
        mock_get_token.return_value = None

        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')

        with pytest.raises(ValueError, match="Not authenticated"):
            upload_dataset(
                dataset_path=str(dataset_file),
                name="Test Dataset",
                slug="test-dataset",
            )

    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_empty_dataset(self, mock_get_token, tmp_path):
        """Test upload with empty dataset file."""
        mock_get_token.return_value = "test-token"

        dataset_file = tmp_path / "empty.jsonl"
        dataset_file.write_text("")

        with pytest.raises(ValueError, match="empty"):
            upload_dataset(
                dataset_path=str(dataset_file),
                name="Empty Dataset",
                slug="empty-dataset",
            )

    @patch("httpx.Client")
    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_successful_upload(self, mock_get_token, mock_client_class, tmp_path):
        """Test successful dataset upload."""
        mock_get_token.return_value = "test-token"

        # Create test dataset
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"messages": [{"role": "user", "content": "test"}]}\n')

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "dataset_id": "new-dataset-id",
            "version_id": "v1",
            "samples_count": 1,
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = upload_dataset(
            dataset_path=str(dataset_file),
            name="Test Dataset",
            slug="test-dataset",
        )

        assert result["dataset_id"] == "new-dataset-id"
        mock_client.post.assert_called_once()


class TestUploadTopicGraph:
    """Tests for upload_topic_graph function."""

    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_not_authenticated(self, mock_get_token, tmp_path):
        """Test upload when not authenticated."""
        mock_get_token.return_value = None

        graph_file = tmp_path / "graph.json"
        graph_file.write_text('{"nodes": {}, "root_id": 1}')

        with pytest.raises(ValueError, match="Not authenticated"):
            upload_topic_graph(
                graph_path=str(graph_file),
                name="Test Graph",
            )

    @patch("httpx.Client")
    @patch("deepfabric.cloud_upload.get_auth_token")
    def test_successful_upload(self, mock_get_token, mock_client_class, tmp_path):
        """Test successful graph upload."""
        mock_get_token.return_value = "test-token"

        # Create test graph
        graph_file = tmp_path / "graph.json"
        graph_file.write_text('{"nodes": {"1": {"id": 1, "topic": "root"}}, "root_id": 1}')

        # Mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "graph": {"id": "new-graph-id", "name": "Test Graph"},
            "nodes_imported": 1,
            "edges_imported": 0,
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        result = upload_topic_graph(
            graph_path=str(graph_file),
            name="Test Graph",
        )

        assert result["nodes_imported"] == 1
        mock_client.post.assert_called_once()


class TestHandleCloudUpload:
    """Tests for handle_cloud_upload function."""

    @patch.dict(os.environ, {"EXPERIMENTAL_DF": ""}, clear=False)
    def test_experimental_flag_not_set(self, tmp_path):
        """Test that upload is skipped when EXPERIMENTAL_DF is not set."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"messages": []}\n')

        result = handle_cloud_upload(
            dataset_path=str(dataset_file),
            graph_path=None,
            cloud_upload_flag="all",
        )
        assert result is None

    @patch.dict(os.environ, {"EXPERIMENTAL_DF": "true"}, clear=False)
    def test_none_flag_skips_in_headless(self, tmp_path):
        """Test that 'none' flag skips upload."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"messages": []}\n')

        result = handle_cloud_upload(
            dataset_path=str(dataset_file),
            graph_path=None,
            cloud_upload_flag="none",
        )
        assert result is None

    @patch.dict(os.environ, {"EXPERIMENTAL_DF": "true"}, clear=False)
    @patch("deepfabric.cloud_upload.ensure_authenticated")
    def test_not_authenticated_headless(self, mock_ensure_auth, tmp_path):
        """Test error when not authenticated in headless mode."""
        mock_ensure_auth.return_value = False

        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text('{"messages": []}\n')

        with patch("deepfabric.cloud_upload.get_tui") as mock_get_tui:
            mock_tui = MagicMock()
            mock_get_tui.return_value = mock_tui

            with pytest.raises(Exception):  # ClickException  # noqa: B017
                handle_cloud_upload(
                    dataset_path=str(dataset_file),
                    graph_path=None,
                    cloud_upload_flag="all",
                )


class TestPromptForName:
    """Tests for prompt_for_name function."""

    @patch("deepfabric.cloud_upload.click.prompt")
    @patch("deepfabric.cloud_upload.get_tui")
    def test_prompt_for_name_basic(self, mock_get_tui, mock_prompt):
        """Test basic name prompting - slug is auto-derived from name."""
        mock_tui = MagicMock()
        mock_get_tui.return_value = mock_tui

        # User enters a custom name (slug is auto-derived, no second prompt)
        mock_prompt.return_value = "My Custom Name"

        name, slug = prompt_for_name("dataset", "Default Name", "default-name")

        assert name == "My Custom Name"
        assert slug == "my-custom-name"  # Auto-derived from name
        assert mock_prompt.call_count == 1  # Only prompted for name

    @patch("deepfabric.cloud_upload.click.prompt")
    @patch("deepfabric.cloud_upload.get_tui")
    def test_prompt_derives_slug_from_name(self, mock_get_tui, mock_prompt):
        """Test that slug is auto-derived and cleaned from name."""
        mock_tui = MagicMock()
        mock_get_tui.return_value = mock_tui

        # User enters name with special characters
        mock_prompt.return_value = "My Name With Spaces!!!"

        name, slug = prompt_for_name("graph", "Default", "default")

        assert name == "My Name With Spaces!!!"
        assert slug == "my-name-with-spaces"  # Auto-cleaned slug
