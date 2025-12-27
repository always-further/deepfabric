"""Integration tests for LLMClient with real API calls."""

import asyncio

import pytest

from deepfabric.llm.client import LLMClient
from deepfabric.schemas import ChatMessage, ChatTranscript, TopicList

from .conftest import requires_gemini, requires_openai


class TestLLMClientOpenAI:
    """Integration tests for LLMClient with OpenAI provider."""

    @requires_openai
    @pytest.mark.openai
    def test_basic_structured_output(self, openai_config):
        """Test basic structured output generation with OpenAI."""
        client = LLMClient(
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
        )

        prompt = "Generate a short greeting conversation between two people."
        result = client.generate(prompt, ChatTranscript)

        assert isinstance(result, ChatTranscript)
        assert len(result.messages) >= 1
        assert all(isinstance(m, ChatMessage) for m in result.messages)

    @requires_openai
    @pytest.mark.openai
    def test_async_structured_output(self, openai_config):
        """Test async structured output generation with OpenAI."""
        client = LLMClient(
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
        )

        async def run_async():
            prompt = "List 3 subtopics about machine learning."
            return await client.generate_async(prompt, TopicList)

        result = asyncio.run(run_async())

        assert isinstance(result, TopicList)
        assert len(result.subtopics) >= 1
        assert all(isinstance(s, str) for s in result.subtopics)

    @requires_openai
    @pytest.mark.openai
    def test_async_streaming(self, openai_config):
        """Test async streaming generation with OpenAI."""
        client = LLMClient(
            provider=openai_config["provider"],
            model_name=openai_config["model_name"],
        )

        async def run_stream():
            prompt = "Generate a brief Q&A about Python programming."
            chunks = []
            final_result = None

            # generate_async_stream yields tuples: (chunk, None) or (None, result)
            async for chunk, result in client.generate_async_stream(prompt, ChatTranscript):
                if chunk is not None:
                    chunks.append(chunk)
                if result is not None:
                    final_result = result

            return chunks, final_result

        chunks, result = asyncio.run(run_stream())

        # Should have received streaming chunks
        assert len(chunks) >= 1
        # Final result should be valid
        assert isinstance(result, ChatTranscript)
        assert len(result.messages) >= 1


class TestLLMClientGemini:
    """Integration tests for LLMClient with Gemini provider.

    Note: Gemini only supports async generation, so all tests use generate_async().
    """

    @requires_gemini
    @pytest.mark.gemini
    def test_basic_structured_output(self, gemini_config):
        """Test basic structured output generation with Gemini."""
        client = LLMClient(
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
        )

        async def run_async():
            prompt = "Generate a short greeting conversation between two people."
            return await client.generate_async(prompt, ChatTranscript)

        result = asyncio.run(run_async())

        assert isinstance(result, ChatTranscript)
        assert len(result.messages) >= 1
        assert all(isinstance(m, ChatMessage) for m in result.messages)

    @requires_gemini
    @pytest.mark.gemini
    def test_async_topic_list(self, gemini_config):
        """Test async structured output generation with Gemini."""
        client = LLMClient(
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
        )

        async def run_async():
            prompt = "List 3 subtopics about data science."
            return await client.generate_async(prompt, TopicList)

        result = asyncio.run(run_async())

        assert isinstance(result, TopicList)
        assert len(result.subtopics) >= 1
        assert all(isinstance(s, str) for s in result.subtopics)

    @requires_gemini
    @pytest.mark.gemini
    def test_gemini_schema_handling(self, gemini_config):
        """Test that Gemini correctly handles schema conversion.

        Gemini has specific requirements around JSON schemas (no additionalProperties,
        specific array constraints). This test verifies the schema conversion works.
        """
        client = LLMClient(
            provider=gemini_config["provider"],
            model_name=gemini_config["model_name"],
        )

        async def run_async():
            # TopicList has min_length constraint which tests array handling
            prompt = "List exactly 2 subtopics about cloud computing."
            return await client.generate_async(prompt, TopicList)

        result = asyncio.run(run_async())

        assert isinstance(result, TopicList)
        assert len(result.subtopics) >= 1
