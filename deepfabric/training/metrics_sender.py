"""Non-blocking async metrics sender for training metrics."""

from __future__ import annotations

import atexit
import logging
import queue
import threading
import time

from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MetricsSender:
    """Non-blocking metrics sender with background thread.

    Queues metrics and sends them in batches via a background thread to avoid
    blocking training. Gracefully handles network errors and queue overflow.

    Example:
        sender = MetricsSender(
            endpoint="https://api.deepfabric.cloud",
            api_key="your-api-key",
        )
        sender.send_metrics({"loss": 2.5, "step": 100})
        sender.flush()  # Ensure all metrics are sent
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str | None,
        pipeline_id: str | None = None,
        batch_size: int = 10,
        flush_interval: float = 5.0,
        max_queue_size: int = 1000,
        timeout: float = 10.0,
    ):
        """Initialize the metrics sender.

        Args:
            endpoint: Base URL for the DeepFabric API
            api_key: API key for authentication (None disables sending)
            pipeline_id: Pipeline ID to associate training runs with (required)
            batch_size: Number of metrics to batch before sending
            flush_interval: Seconds between automatic flushes
            max_queue_size: Maximum queue size (overflow drops metrics)
            timeout: HTTP request timeout in seconds
        """
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.timeout = timeout

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._flush_event = threading.Event()
        self._enabled = api_key is not None

        # Start background sender thread
        if self._enabled:
            self._thread = threading.Thread(
                target=self._sender_loop,
                daemon=True,
                name="deepfabric-metrics-sender",
            )
            self._thread.start()
            atexit.register(self.shutdown)
        else:
            self._thread = None

        self._send_errors = 0
        self._metrics_sent = 0
        self._metrics_dropped = 0

    @property
    def enabled(self) -> bool:
        """Whether the sender is enabled (has API key)."""
        return self._enabled

    @property
    def stats(self) -> dict[str, int]:
        """Get sender statistics."""
        return {
            "metrics_sent": self._metrics_sent,
            "metrics_dropped": self._metrics_dropped,
            "send_errors": self._send_errors,
            "queue_size": self._queue.qsize(),
        }

    def send_metrics(self, metrics: dict[str, Any]) -> bool:
        """Queue metrics for async sending (non-blocking).

        Args:
            metrics: Dictionary of metric names to values

        Returns:
            True if queued successfully, False if dropped
        """
        if not self._enabled:
            return False

        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            self._queue.put_nowait({"type": "metrics", "data": metrics})
        except queue.Full:
            self._metrics_dropped += 1
            logger.debug("Metrics queue full, dropping metrics")
            return False
        else:
            return True

    def send_run_start(self, metadata: dict[str, Any]) -> bool:
        """Send run start event.

        Args:
            metadata: Run metadata (model_name, training_args, etc.)

        Returns:
            True if queued successfully
        """
        return self._send_event("run_start", metadata)

    def send_run_end(self, metadata: dict[str, Any]) -> bool:
        """Send run end event.

        Args:
            metadata: Run end metadata (final_step, final_epoch, etc.)

        Returns:
            True if queued successfully
        """
        return self._send_event("run_end", metadata)

    def _send_event(self, event_type: str, data: dict[str, Any]) -> bool:
        """Queue an event for sending.

        Args:
            event_type: Type of event (run_start, run_end, etc.)
            data: Event data

        Returns:
            True if queued successfully
        """
        if not self._enabled:
            return False

        if "timestamp" not in data:
            data["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            self._queue.put_nowait({"type": event_type, "data": data})
        except queue.Full:
            logger.debug(f"Queue full, dropping {event_type} event")
            return False
        else:
            return True

    def _sender_loop(self) -> None:
        """Background thread that batches and sends metrics."""
        batch: list[dict[str, Any]] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            try:
                # Wait for item with timeout
                item = self._queue.get(timeout=min(1.0, self.flush_interval))
                batch.append(item)

                # Check if we should flush
                should_flush = (
                    len(batch) >= self.batch_size
                    or (time.monotonic() - last_flush) >= self.flush_interval
                    or self._flush_event.is_set()
                )

                if should_flush:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.monotonic()
                    self._flush_event.clear()

            except queue.Empty:
                # Timeout - flush if we have pending items or flush requested
                if batch and (
                    (time.monotonic() - last_flush) >= self.flush_interval
                    or self._flush_event.is_set()
                ):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.monotonic()
                    self._flush_event.clear()

        # On shutdown, drain the queue and flush everything
        while not self._queue.empty():
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[dict[str, Any]]) -> None:
        """Send batch of metrics to API.

        Args:
            batch: List of queued items to send
        """
        if not batch or not self._enabled:
            return

        if not self.pipeline_id:
            logger.debug("No pipeline_id set, skipping metrics send")
            return

        # Separate events and metrics
        run_start_events = [item for item in batch if item["type"] == "run_start"]
        run_end_events = [item for item in batch if item["type"] == "run_end"]
        metrics = [item["data"] for item in batch if item["type"] == "metrics"]

        # Build query string with pipeline_id
        query = f"?pipeline_id={self.pipeline_id}"

        def send_run_events(events: list[dict[str, Any]]) -> None:
            """Send run start/end events."""
            for event in events:
                self._send_to_api(
                    endpoint=f"{self.endpoint}/api/v1/training/runs{query}",
                    payload={"event_type": event["type"], **event["data"]},
                )

        # Send run events, ensuring start events are processed before end events
        send_run_events(run_start_events)
        send_run_events(run_end_events)

        # Send metrics batch
        if metrics:
            self._send_to_api(
                endpoint=f"{self.endpoint}/api/v1/training/metrics{query}",
                payload={"metrics": metrics},
            )
            self._metrics_sent += len(metrics)

    def _send_to_api(self, endpoint: str, payload: dict[str, Any]) -> bool:
        """Send payload to API endpoint.

        Args:
            endpoint: Full API endpoint URL
            payload: JSON payload to send

        Returns:
            True if sent successfully
        """
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "deepfabric-training/1.0",
                },
                timeout=self.timeout,
            )

            if not response.ok:
                self._send_errors += 1
                logger.warning(
                    "API error: %s %s (endpoint: %s)",
                    response.status_code,
                    response.text[:200],
                    endpoint,
                )
                return False

        except requests.exceptions.Timeout:
            self._send_errors += 1
            logger.warning("Request timed out: %s", endpoint)
            return False

        except requests.exceptions.ConnectionError as e:
            self._send_errors += 1
            logger.warning("Connection error: %s (endpoint: %s)", e, endpoint)
            return False

        except requests.exceptions.RequestException as e:
            self._send_errors += 1
            logger.warning("Request error: %s (endpoint: %s)", e, endpoint)
            return False

        else:
            return True

    def flush(self, timeout: float = 30.0) -> None:
        """Flush all pending metrics (blocking).

        Args:
            timeout: Maximum time to wait for flush
        """
        if not self._enabled:
            return

        # Signal the background thread to flush its current batch
        self._flush_event.set()

        start = time.monotonic()
        # Wait for queue to empty and flush event to be cleared (indicates batch was sent)
        while (time.monotonic() - start) < timeout:
            if self._queue.empty() and not self._flush_event.is_set():
                break
            time.sleep(0.1)

    def shutdown(self) -> None:
        """Stop the sender thread and flush remaining metrics."""
        if not self._enabled or self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)

        # Log final stats
        logger.debug(
            f"MetricsSender shutdown: sent={self._metrics_sent}, "
            f"dropped={self._metrics_dropped}, errors={self._send_errors}"
        )
