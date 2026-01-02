from .auth import (
    clear_tokens,
    device_flow_login,
    get_auth_token,
    get_config,
    get_stored_token,
    is_authenticated,
    prompt_cloud_signup,
    save_config,
    store_tokens,
)
from .cli import cli
from .config import DeepFabricConfig
from .dataset import Dataset, DatasetDict
from .exceptions import (
    APIError,
    ConfigurationError,
    DatasetError,
    DataSetGeneratorError,
    DeepFabricError,
    HubUploadError,
    JSONParsingError,
    LoaderError,
    ModelError,
    RetryExhaustedError,
    TreeError,
    ValidationError,
)
from .generator import DataSetGenerator, DataSetGeneratorConfig
from .graph import Graph, GraphConfig
from .hf_hub import HFUploader
from .loader import load_dataset
from .training import DeepFabricCallback, MetricsSender
from .tree import Tree, TreeConfig

__version__ = "0.1.0"

__all__ = [
    "Tree",
    "TreeConfig",
    "Graph",
    "GraphConfig",
    "DataSetGenerator",
    "DataSetGeneratorConfig",
    "DeepFabricConfig",
    "HFUploader",
    "cli",
    # Dataset loading
    "load_dataset",
    "Dataset",
    "DatasetDict",
    # Training metrics logging
    "DeepFabricCallback",
    "MetricsSender",
    # Authentication
    "get_config",
    "save_config",
    "get_stored_token",
    "store_tokens",
    "clear_tokens",
    "is_authenticated",
    "get_auth_token",
    "prompt_cloud_signup",
    "device_flow_login",
    # Exceptions
    "DeepFabricError",
    "ConfigurationError",
    "ValidationError",
    "ModelError",
    "TreeError",
    "DataSetGeneratorError",
    "DatasetError",
    "HubUploadError",
    "JSONParsingError",
    "APIError",
    "RetryExhaustedError",
    "LoaderError",
]
