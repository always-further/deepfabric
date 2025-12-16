"""Tests for Gemini schema transformation functions and DirectGeminiModel."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from deepfabric.llm.client import (
    AsyncDirectGeminiModel,
    DirectGeminiModel,
    _inline_refs,
    _strip_additional_properties,
)


class SubItem(BaseModel):
    """A nested model for testing $ref handling."""

    name: str
    value: int


class ParentItem(BaseModel):
    """A parent model with nested references."""

    title: str
    sub: SubItem
    items: list[SubItem]


class ModelWithDict(BaseModel):
    """Model with dict[str, Any] field that Gemini doesn't support."""

    name: str
    metadata: dict  # This becomes additionalProperties: true


class ModelWithOptionalDict(BaseModel):
    """Model with optional dict field."""

    name: str
    extra: dict | None = None


class TestInlineRefs:
    """Tests for _inline_refs function."""

    def test_simple_schema_unchanged(self):
        """Simple schema without $ref should remain unchanged."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        result = _inline_refs(schema)
        assert result == schema

    def test_inlines_ref_from_defs(self):
        """$ref references should be inlined from $defs."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/SubItem"},
            },
            "$defs": {
                "SubItem": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                }
            },
        }
        result = _inline_refs(schema)

        # $ref should be inlined
        assert "$ref" not in result["properties"]["item"]
        assert result["properties"]["item"]["type"] == "object"
        assert "name" in result["properties"]["item"]["properties"]

        # $defs should be removed from result
        assert "$defs" not in result

    def test_inlines_nested_refs(self):
        """Nested $ref references should be recursively inlined."""
        schema = {
            "type": "object",
            "properties": {
                "outer": {"$ref": "#/$defs/Outer"},
            },
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                },
                "Outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"$ref": "#/$defs/Inner"},
                    },
                },
            },
        }
        result = _inline_refs(schema)

        # Both refs should be inlined
        outer = result["properties"]["outer"]
        assert "$ref" not in outer
        inner = outer["properties"]["inner"]
        assert "$ref" not in inner
        assert inner["properties"]["value"]["type"] == "integer"

    def test_inlines_refs_in_array_items(self):
        """$ref in array items should be inlined."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Item"},
                }
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        result = _inline_refs(schema)

        items_schema = result["properties"]["items"]["items"]
        assert "$ref" not in items_schema
        assert items_schema["type"] == "object"

    def test_inlines_refs_in_anyof(self):
        """$ref in anyOf variants should be inlined."""
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"$ref": "#/$defs/TypeA"},
                        {"type": "null"},
                    ]
                }
            },
            "$defs": {
                "TypeA": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                }
            },
        }
        result = _inline_refs(schema)

        anyof = result["properties"]["value"]["anyOf"]
        assert "$ref" not in anyof[0]
        assert anyof[0]["type"] == "object"

    def test_preserves_sibling_properties(self):
        """Properties alongside $ref should be preserved after inlining."""
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "$ref": "#/$defs/Item",
                    "description": "An item",
                }
            },
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        result = _inline_refs(schema)

        item = result["properties"]["item"]
        assert item["description"] == "An item"
        assert item["type"] == "object"

    def test_handles_non_dict_input(self):
        """Non-dict input should be returned as-is."""
        assert _inline_refs("string") == "string"
        assert _inline_refs(123) == 123
        assert _inline_refs(None) is None


class TestStripAdditionalProperties:
    """Tests for _strip_additional_properties function."""

    def test_removes_additional_properties_field(self):
        """additionalProperties field should be removed."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        result = _strip_additional_properties(schema)
        assert "additionalProperties" not in result

    def test_removes_dict_fields(self):
        """Fields with additionalProperties: true should be removed."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "metadata": {"type": "object", "additionalProperties": True},
            },
            "required": ["name", "metadata"],
        }
        result = _strip_additional_properties(schema)

        assert "name" in result["properties"]
        assert "metadata" not in result["properties"]
        assert "metadata" not in result["required"]

    def test_removes_object_without_properties(self):
        """Object type without properties (like dict[str, Any]) should be removed."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "data": {"type": "object"},  # No properties = incompatible
            },
            "required": ["name", "data"],
        }
        result = _strip_additional_properties(schema)

        assert "name" in result["properties"]
        assert "data" not in result["properties"]

    def test_removes_incompatible_array_items(self):
        """Arrays with incompatible object items should be removed."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "items": {"type": "object"},  # No properties
                },
            },
            "required": ["name"],
        }
        result = _strip_additional_properties(schema)

        assert "name" in result["properties"]
        assert "items" not in result["properties"]

    def test_processes_nested_schemas(self):
        """Nested schemas should be recursively processed."""
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "string"},
                    },
                    "additionalProperties": False,
                }
            },
        }
        result = _strip_additional_properties(schema)

        assert "additionalProperties" not in result["properties"]["outer"]

    def test_processes_defs(self):
        """$defs should be recursively processed."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "additionalProperties": False,
                }
            },
        }
        result = _strip_additional_properties(schema)

        assert "additionalProperties" not in result["$defs"]["Inner"]

    def test_handles_anyof_with_incompatible_variants(self):
        """anyOf containing incompatible variants should remove the field."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "data": {
                    "anyOf": [
                        {"type": "object"},  # Incompatible
                        {"type": "null"},
                    ]
                },
            },
            "required": ["name"],
        }
        result = _strip_additional_properties(schema)

        assert "data" not in result["properties"]


class TestDirectGeminiModel:
    """Tests for DirectGeminiModel class."""

    def test_prepare_json_schema_inlines_and_strips(self):
        """_prepare_json_schema should inline refs and strip incompatible fields."""
        mock_client = MagicMock()
        model = DirectGeminiModel(mock_client, "gemini-1.5-flash")

        # Use a real Pydantic model
        schema = model._prepare_json_schema(ParentItem)

        # Should not have $ref (inlined)
        assert "$defs" not in schema
        # Check structure is preserved
        assert "properties" in schema
        assert "title" in schema["properties"]
        assert "sub" in schema["properties"]

    def test_prepare_json_schema_removes_dict_fields(self):
        """_prepare_json_schema should remove dict[str, Any] fields."""
        mock_client = MagicMock()
        model = DirectGeminiModel(mock_client, "gemini-1.5-flash")

        schema = model._prepare_json_schema(ModelWithDict)

        # metadata field should be removed (has additionalProperties)
        assert "name" in schema["properties"]
        assert "metadata" not in schema["properties"]

    @patch("deepfabric.llm.client.genai_types.GenerateContentConfig")
    def test_call_uses_response_json_schema(self, mock_config_class):
        """__call__ should use response_json_schema parameter."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"title": "Test", "sub": {"name": "Sub", "value": 1}, "items": []}'
        mock_client.models.generate_content.return_value = mock_response

        model = DirectGeminiModel(mock_client, "gemini-1.5-flash")
        result = model("Generate something", ParentItem)

        # Verify config was created with response_json_schema
        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args.kwargs
        assert call_kwargs["response_mime_type"] == "application/json"
        assert "response_json_schema" in call_kwargs

        # Verify result
        assert result == mock_response.text

    def test_call_raises_on_empty_response(self):
        """__call__ should raise DataSetGeneratorError on empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.models.generate_content.return_value = mock_response

        model = DirectGeminiModel(mock_client, "gemini-1.5-flash")

        from deepfabric.exceptions import DataSetGeneratorError

        with pytest.raises(DataSetGeneratorError, match="empty response"):
            model("Generate something", ParentItem)


class TestAsyncDirectGeminiModel:
    """Tests for AsyncDirectGeminiModel class."""

    def test_prepare_json_schema_same_as_sync(self):
        """Async model should prepare schemas the same as sync model."""
        mock_client = MagicMock()
        sync_model = DirectGeminiModel(mock_client, "gemini-1.5-flash")
        async_model = AsyncDirectGeminiModel(mock_client, "gemini-1.5-flash")

        sync_schema = sync_model._prepare_json_schema(ParentItem)
        async_schema = async_model._prepare_json_schema(ParentItem)

        assert sync_schema == async_schema

    @patch("deepfabric.llm.client.genai_types.GenerateContentConfig")
    def test_call_uses_async_api(self, mock_config_class):
        """__call__ should use async API (client.aio.models.generate_content)."""
        import asyncio

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"title": "Test", "sub": {"name": "Sub", "value": 1}, "items": []}'

        # Set up async mock
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        model = AsyncDirectGeminiModel(mock_client, "gemini-1.5-flash")
        result = asyncio.run(model("Generate something", ParentItem))

        # Verify async API was called
        mock_client.aio.models.generate_content.assert_called_once()

        # Verify result
        assert result == mock_response.text

    def test_call_raises_on_empty_response(self):
        """__call__ should raise DataSetGeneratorError on empty response."""
        import asyncio

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = None
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        model = AsyncDirectGeminiModel(mock_client, "gemini-1.5-flash")

        from deepfabric.exceptions import DataSetGeneratorError

        with pytest.raises(DataSetGeneratorError, match="empty response"):
            asyncio.run(model("Generate something", ParentItem))


class TestPydanticModelSchemas:
    """Integration tests using real Pydantic model schemas."""

    def test_parent_item_schema_transformation(self):
        """ParentItem schema should be properly transformed for Gemini."""
        original = ParentItem.model_json_schema()

        # Original should have $defs
        assert "$defs" in original
        assert "SubItem" in original["$defs"]

        # Transform
        transformed = _strip_additional_properties(_inline_refs(original))

        # Transformed should not have $defs
        assert "$defs" not in transformed

        # SubItem should be inlined in both 'sub' and 'items'
        sub_schema = transformed["properties"]["sub"]
        assert sub_schema["type"] == "object"
        assert "name" in sub_schema["properties"]
        assert "value" in sub_schema["properties"]

        items_schema = transformed["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "name" in items_schema["properties"]

    def test_model_with_connections_field(self):
        """Test schema similar to GraphSubtopic with connections field."""

        class GraphSubtopic(BaseModel):
            topic: str = Field(description="The subtopic name")
            connections: list[str] = Field(
                default_factory=list,
                description="Related topics",
            )

        original = GraphSubtopic.model_json_schema()
        transformed = _strip_additional_properties(_inline_refs(original))

        # Should preserve the structure
        assert "topic" in transformed["properties"]
        assert "connections" in transformed["properties"]
        assert transformed["properties"]["connections"]["type"] == "array"
