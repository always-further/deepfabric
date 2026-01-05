import json
import logging
import sys

from functools import cached_property
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ...schemas import ToolDefinition
from ...utils import import_optional_dependency
from ..inference import InferenceBackend, InferenceConfig, ModelResponse
from .tool_call_parsers import ToolCallParser, get_parser

# Mistral-family architectures that require fix_mistral_regex=True
MISTRAL_ARCHITECTURES = frozenset(
    {
        "MistralForCausalLM",
        "Mistral3ForConditionalGeneration",
        "MixtralForCausalLM",
        "MinistralForCausalLM",
        "PixtralForConditionalGeneration",
    }
)

logger = logging.getLogger(__name__)


class TransformersBackend(InferenceBackend):
    """Inference backend using HuggingFace Transformers."""

    @cached_property
    def _torch(self) -> Any:
        """Dynamically import 'torch' and verify its availability.

        Returns:
            The imported torch module.

        Raises:
            ModuleNotFoundError: If 'torch' is not installed in the environment.
        """
        return import_optional_dependency("torch", "training")

    @cached_property
    def _peft(self) -> Any:
        """Dynamically import 'peft' and verify its availability.

        Returns:
            The imported peft module.

        Raises:
            ModuleNotFoundError: If 'peft' is not installed in the environment.
        """
        return import_optional_dependency("peft", "training")

    def __init__(self, config: InferenceConfig):
        """Initialize Transformers backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)

        # Check if model is pre-loaded (not a string path)
        is_preloaded = not isinstance(config.model, str)

        # Determine device
        if config.device:
            self.device = config.device
        elif is_preloaded:
            # Get device from pre-loaded model
            self.device = str(next(config.model.parameters()).device)
        # Auto-detect best available device
        elif self._torch.cuda.is_available():
            self.device = "cuda"
        elif self._torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Determine dtype based on device
        if self.device == "cuda" or self.device.startswith("cuda:"):
            dtype = self._torch.float16
            device_map = "auto"
        elif self.device == "mps":
            dtype = self._torch.float32  # MPS works best with float32
            device_map = None
        else:
            dtype = self._torch.float32
            device_map = None

        # Handle pre-loaded model case - skip all loading logic
        if is_preloaded:
            self.model = config.model
            self.tokenizer = config.tokenizer
            self.loaded_with_unsloth = False

            # Detect architecture from pre-loaded model's config
            self._architectures = []
            if hasattr(self.model, "config"):
                self._architectures = getattr(self.model.config, "architectures", []) or []

            # Initialize tool call parser
            self._tool_call_parser: ToolCallParser = get_parser(self._architectures)
            logger.info(
                "Using pre-loaded model with %s parser for architectures: %s",
                type(self._tool_call_parser).__name__,
                self._architectures or ["unknown"],
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            return  # Skip remaining initialization

        # Detect model architecture for parser selection and tokenizer config
        self._architectures = []
        tokenizer_kwargs: dict[str, Any] = {}
        try:
            model_config = AutoConfig.from_pretrained(config.model)  # nosec
            self._architectures = getattr(model_config, "architectures", []) or []
            if any(arch in MISTRAL_ARCHITECTURES for arch in self._architectures):
                tokenizer_kwargs["fix_mistral_regex"] = True
                logger.debug("Detected Mistral architecture, enabling fix_mistral_regex")
        except Exception as e:
            logger.warning("Could not detect model architecture: %s", e)

        # Initialize tool call parser based on detected architecture
        self._tool_call_parser = get_parser(self._architectures)
        logger.info(
            "Using %s for model architectures: %s",
            type(self._tool_call_parser).__name__,
            self._architectures or ["unknown"],
        )

        self.loaded_with_unsloth = False

        # Detect if Unsloth has already patched the environment
        # This happens when user imports unsloth in the same runtime
        unsloth_patched = "unsloth" in sys.modules

        # Use Unsloth if explicitly requested OR if Unsloth has patched the environment
        # (to avoid "apply_qkv" errors from patched attention classes)
        use_unsloth_loading = config.use_unsloth or unsloth_patched

        if use_unsloth_loading:
            try:
                from unsloth import FastLanguageModel  # type: ignore # noqa: PLC0415

                if unsloth_patched and not config.use_unsloth:
                    logger.info(
                        "Unsloth detected in environment, using Unsloth loader for compatibility"
                    )

                if config.adapter_path:
                    # Load base model first, then apply adapter
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config.model,
                        max_seq_length=config.max_seq_length,
                        dtype=dtype,
                        load_in_4bit=config.load_in_4bit,
                    )
                    # Load LoRA adapter using PEFT
                    self.model = self._peft.PeftModel.from_pretrained(
                        self.model, config.adapter_path
                    )
                else:
                    # Load merged model or base model directly
                    self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                        model_name=config.model,
                        max_seq_length=config.max_seq_length,
                        dtype=dtype,
                        load_in_4bit=config.load_in_4bit,
                    )
                FastLanguageModel.for_inference(self.model)
                self.loaded_with_unsloth = True
            except ImportError:
                logger.warning("Unsloth not installed, falling back to standard transformers")
            except Exception as e:
                logger.warning(
                    "Unsloth loading failed (%s), falling back to standard transformers", e
                )

        # Standard transformers/PEFT loading
        if not self.loaded_with_unsloth:
            self.tokenizer = AutoTokenizer.from_pretrained(  # nosec
                config.model, **tokenizer_kwargs
            )

            self.model = AutoModelForCausalLM.from_pretrained(  # nosec
                config.model,
                device_map=device_map,
                dtype=dtype,
            )

            # Load PEFT adapter if provided
            if config.adapter_path:
                self.model = self._peft.PeftModel.from_pretrained(self.model, config.adapter_path)

            # Move to device if not using device_map
            if self.device in ("cpu", "mps"):
                self.model.to(self.device)  # type: ignore[arg-type]

            # Note: torch.compile disabled - causes very slow first inference
            # due to CUDA graph compilation overhead. For evaluation workloads
            # with many short inferences, the compilation cost isn't amortized.
            # Uncomment for long-running inference servers:
            # with suppress(Exception):
            #     self.model = torch.compile(self.model, mode="reduce-overhead")

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response from model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """
        # Format messages using chat template
        prompt = self._format_prompt(messages, tools)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate with optimizations
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Performance optimizations
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding (faster than beam search)
            )

        # Decode output
        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse tool calls if present
        tool_calls = self._tool_call_parser.parse(generated_text) if tools else []
        tool_call = tool_calls[0] if tool_calls else None

        return ModelResponse(
            content=generated_text,
            tool_call=tool_call,
            tool_calls=tool_calls if tool_calls else None,
            raw_output=generated_text,
            finish_reason="stop",
        )

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of message sequences.

        Args:
            batch_messages: List of message sequences
            tools: Optional list of available tools for function calling

        Returns:
            List of ModelResponse objects
        """
        # Format all prompts
        prompts = [self._format_prompt(msgs, tools) for msgs in batch_messages]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate batch with optimizations
        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Performance optimizations
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding (faster than beam search)
            )

        # Decode outputs
        responses = []
        for i, output_ids in enumerate(outputs):
            # Extract generated portion (skip input tokens)
            generated_ids = output_ids[inputs.input_ids[i].shape[0] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse tool calls if present
            tool_calls = self._tool_call_parser.parse(generated_text) if tools else []
            tool_call = tool_calls[0] if tool_calls else None

            responses.append(
                ModelResponse(
                    content=generated_text,
                    tool_call=tool_call,
                    tool_calls=tool_calls if tool_calls else None,
                    raw_output=generated_text,
                    finish_reason="stop",
                )
            )

        return responses

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _format_prompt(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> str:
        """Format messages into a prompt string.

        Args:
            messages: List of message dicts
            tools: Optional list of tools

        Returns:
            Formatted prompt string
        """
        # Try to use chat template with tools support (modern approach)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                # Convert tools to OpenAI format for chat template compatibility
                tools_param = None
                if tools:
                    tools_param = [tool.to_openai() for tool in tools]

                # Try with tools parameter (for models with native tool support)
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools_param,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, KeyError):
                # Model's chat template doesn't support tools parameter
                # Try without tools parameter
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:  # noqa: S110
                    # Fallback to manual formatting
                    pass  # nosec

        # Manual formatting fallback (for models without chat templates)
        prompt_parts = []

        # Add tools if present
        if tools:
            tools_str = "Available tools:\n"
            for tool in tools:
                tools_str += f"- {tool.name}: {tool.description}\n"
                params_list = [p.model_dump() for p in tool.parameters]
                tools_str += f"  Parameters: {json.dumps(params_list)}\n"
            prompt_parts.append(tools_str)

        # Add messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"{role.upper()}: {content}")

        prompt_parts.append("ASSISTANT:")
        return "\n\n".join(prompt_parts)
