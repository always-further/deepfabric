"""Cloud-based reporter for sending results to DeepFabric Cloud."""

from __future__ import annotations

import json
import os

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from rich.console import Console

from ...utils import get_bool_env
from .base import BaseReporter

if TYPE_CHECKING:
    from ..evaluator import EvaluationResult
    from ..metrics import SampleEvaluation

console = Console()


def get_auth_token() -> str | None:
    """Get authentication token from CLI config."""
    config_file = Path.home() / ".deepfabric" / "config.json"
    if not config_file.exists():
        return None

    try:
        with open(config_file) as f:
            config = json.load(f)
            # Return API key if present, otherwise access token
            return config.get("api_key") or config.get("access_token")
    except (json.JSONDecodeError, OSError):
        return None


class CloudReporter(BaseReporter):
    """Posts evaluation results to DeepFabric cloud service."""

    def __init__(self, config: dict | None = None):
        """Initialize cloud reporter.

        Args:
            config: Optional configuration with:
                - api_url: DeepFabric API URL (default: https://api.deepfabric.cloud")
                - project_id: Project ID to associate results with
                - auth_token: Authentication token (if not provided, will read from config file)
                - enabled: Whether to enable cloud reporting (default: True if authenticated)
        """
        super().__init__(config)

        # Get API URL from config or environment
        self.api_url = os.getenv("DEEPFABRIC_API_URL", "https://api.deepfabric.cloud")
        if config and "api_url" in config:
            self.api_url = config["api_url"]

        # Get auth token from config or CLI config file
        self.auth_token = None
        if config and "auth_token" in config:
            self.auth_token = config["auth_token"]
        else:
            self.auth_token = get_auth_token()

        # Get project ID from config
        self.project_id = config.get("project_id") if config else None

        # Enable cloud reporting if authenticated AND experimental flag is set
        is_experimental = get_bool_env("EXPERIMENTAL_DF")
        self.enabled = is_experimental and (
            config.get("enabled", bool(self.auth_token)) if config else bool(self.auth_token)
        )

        # Generate unique run ID for this evaluation
        self.run_id = None  # Will be set when creating run
        self.evaluation_run_id = None  # Backend run ID

    def report(self, result: EvaluationResult) -> None:
        """Upload complete evaluation results to cloud service.

        Args:
            result: Complete evaluation result
        """
        if not self.enabled:
            return

        if not self.auth_token:
            console.print(
                "[yellow]Cloud reporting skipped: Not authenticated. "
                "Run 'deepfabric auth login' to enable cloud sync.[/yellow]"
            )
            return

        if not self.project_id:
            console.print("[yellow]Cloud reporting skipped: No project_id configured.[/yellow]")
            return

        try:
            console.print("[cyan]Uploading evaluation results to cloud...[/cyan]")

            # Get model name as string (handle in-memory model objects)
            model_value = result.config.inference_config.model
            if isinstance(model_value, str):
                model_name = model_value
            else:
                # For in-memory model objects, extract name from config
                model_config = getattr(model_value, "config", None)
                model_name = (
                    getattr(model_config, "name_or_path", None) or type(model_value).__name__
                )

            # Create evaluation run
            run_data = {
                "pipeline_id": self.project_id,
                "name": f"Evaluation - {datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}",
                "model_name": model_name,
                "model_provider": result.config.inference_config.backend,
                "config": {
                    "evaluators": getattr(result.config, "evaluators", ["tool_calling"]),
                    "inference": result.config.inference_config.model_dump(),
                },
                "status": "completed",
            }

            with httpx.Client(timeout=30.0) as client:
                # Create run
                response = client.post(
                    f"{self.api_url}/api/v1/evaluations/runs",
                    json=run_data,
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                run_response = response.json()
                self.evaluation_run_id = run_response["id"]

                console.print(f"[green]v[/green] Created evaluation run: {self.evaluation_run_id}")

                # Upload metrics
                metrics_data = {
                    "overall_score": result.metrics.overall_score,
                    "tool_selection_accuracy": result.metrics.tool_selection_accuracy,
                    "parameter_accuracy": result.metrics.parameter_accuracy,
                    "execution_success_rate": result.metrics.execution_success_rate,
                    "response_quality": result.metrics.response_quality,
                    "samples_evaluated": result.metrics.samples_evaluated,
                    "samples_processed": result.metrics.samples_processed,
                    "processing_errors": result.metrics.processing_errors,
                }

                response = client.post(
                    f"{self.api_url}/api/v1/evaluations/runs/{self.evaluation_run_id}/metrics",
                    json=metrics_data,
                    headers={
                        "Authorization": f"Bearer {self.auth_token}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()

                console.print("[green]v[/green] Uploaded metrics")

                # Upload samples in batches
                batch_size = 100
                samples = []
                for s in result.predictions:
                    sample_dict = s.model_dump()
                    # Convert sample_id to string (backend expects str, CLI uses int)
                    sample_dict["sample_id"] = str(sample_dict["sample_id"])
                    samples.append(sample_dict)

                for i in range(0, len(samples), batch_size):
                    batch = samples[i : i + batch_size]
                    response = client.post(
                        f"{self.api_url}/api/v1/evaluations/runs/{self.evaluation_run_id}/samples",
                        json={"samples": batch},
                        headers={
                            "Authorization": f"Bearer {self.auth_token}",
                            "Content-Type": "application/json",
                        },
                    )
                    response.raise_for_status()

                console.print(f"[green]v[/green] Uploaded {len(samples)} samples")
                console.print("[green]Results uploaded successfully![/green]")
                console.print(
                    f"View at: {self.api_url.replace(':8080', ':3000')}/studio/evaluations/{self.evaluation_run_id}"
                )

        except httpx.HTTPError as e:
            console.print(f"[red]Cloud upload failed: {e}[/red]")
        except Exception as e:
            console.print(f"[red]Cloud upload error: {e}[/red]")

    def report_sample(self, sample_eval: SampleEvaluation) -> None:  # noqa: ARG002
        """Stream individual sample to cloud for real-time progress tracking.

        Args:
            sample_eval: Individual sample evaluation result
        """
        # Real-time streaming not implemented yet
        # Samples are uploaded in batch in report()
        pass
