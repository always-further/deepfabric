from abc import ABC, abstractmethod
from typing import NamedTuple


class TopicPath(NamedTuple):
    """A topic path with its associated unique identifier."""

    path: list[str]
    topic_id: str


class TopicModel(ABC):
    """Abstract base class for topic models like Tree and Graph."""

    @abstractmethod
    async def build_async(self) -> None:
        """Asynchronously build the topic model."""
        raise NotImplementedError

    def build(self) -> None:  # pragma: no cover - legacy compatibility
        """Deprecated synchronous entry point kept for legacy compatibility."""
        msg = "TopicModel.build() is no longer supported. Use build_async() instead."
        raise RuntimeError(msg)

    @abstractmethod
    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        raise NotImplementedError

    @abstractmethod
    def get_all_paths_with_ids(self) -> list[TopicPath]:
        """Returns all paths with their unique identifiers.

        Returns:
            List of TopicPath namedtuples containing (path, topic_id).
            The topic_id is a stable identifier for the leaf node of each path.
        """
        raise NotImplementedError

    def get_path_by_id(self, topic_id: str) -> list[str] | None:
        """Look up a path by its topic_id.

        Args:
            topic_id: The unique identifier for a topic path.

        Returns:
            The path list if found, None otherwise.
        """
        for topic_path in self.get_all_paths_with_ids():
            if topic_path.topic_id == topic_id:
                return topic_path.path
        return None
