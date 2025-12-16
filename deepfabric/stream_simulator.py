"""Buffered stream simulation for TUI preview.

This module provides a fire-and-forget streaming simulation that emits chunks
to the TUI preview without impacting generation performance. The simulation
runs in background while generation continues immediately.
"""

import asyncio

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from .constants import STREAM_SIM_CHUNK_DELAY_MS, STREAM_SIM_CHUNK_SIZE

if TYPE_CHECKING:
    from .progress import ProgressReporter


class StreamSimulatorConfig(BaseModel):
    """Configuration for buffered stream simulation."""

    chunk_size: int = Field(default=STREAM_SIM_CHUNK_SIZE, ge=1, le=100)
    chunk_delay_ms: float = Field(default=STREAM_SIM_CHUNK_DELAY_MS, ge=0.0, le=100.0)
    enabled: bool = Field(default=True)


class StreamSimulator:
    """Simulates streaming by buffering completed text and emitting chunks.

    Uses fire-and-forget pattern for zero performance impact on generation.
    The simulation runs in background while the caller continues immediately.
    """

    def __init__(
        self,
        progress_reporter: "ProgressReporter | None",
        config: StreamSimulatorConfig | None = None,
    ) -> None:
        self._reporter = progress_reporter
        self._config = config or StreamSimulatorConfig()
        self._active_tasks: set[asyncio.Task] = set()

    async def _simulate_impl(self, content: str, source: str, **metadata) -> None:
        """Internal implementation of chunk emission.

        Args:
            content: Text to simulate streaming
            source: Source identifier for TUI routing
            **metadata: Additional metadata passed to emit_chunk
        """
        if not content:
            return

        delay = self._config.chunk_delay_ms / 1000.0
        chunk_size = self._config.chunk_size

        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            self._reporter.emit_chunk(source, chunk, **metadata)
            if delay > 0 and i + chunk_size < len(content):
                await asyncio.sleep(delay)

    def simulate(self, content: str, source: str, **metadata) -> asyncio.Task | None:
        """Start stream simulation as background task (fire-and-forget).

        Returns immediately. Simulation runs in background without blocking.

        Args:
            content: Text to simulate streaming
            source: Source identifier for TUI routing
            **metadata: Additional metadata passed to emit_chunk

        Returns:
            Task if started, None if no-op (no reporter or disabled)
        """
        if not self._reporter or not self._config.enabled:
            return None

        task = asyncio.create_task(self._simulate_impl(content, source, **metadata))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)
        return task


def simulate_stream(
    progress_reporter: "ProgressReporter | None",
    content: str,
    source: str,
    config: StreamSimulatorConfig | None = None,
    **metadata,
) -> asyncio.Task | None:
    """Fire-and-forget stream simulation (non-blocking).

    Starts simulation in background and returns immediately. This is the
    primary interface for stream simulation throughout the codebase.

    Args:
        progress_reporter: ProgressReporter instance or None
        content: Text to simulate streaming
        source: Source identifier for TUI routing
        config: Optional StreamSimulatorConfig override
        **metadata: Additional metadata passed to emit_chunk

    Returns:
        Task if started, None if no-op (no reporter or disabled)
    """
    simulator = StreamSimulator(progress_reporter, config)
    return simulator.simulate(content, source, **metadata)
