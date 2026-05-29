"""
Experiment Tracker
==================
Lightweight experiment tracking with MLflow-compatible logging interface.
Tracks hyperparameters, metrics, artifacts, and model versions across runs.

Provides:
  - Run management (create, resume, compare)
  - Metric logging with step tracking
  - Artifact storage
  - JSON-based persistence (no server required)
  - Optional MLflow backend integration
"""

import json
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Run:
    """Single experiment run record."""
    run_id: str
    experiment_name: str
    status: str = "RUNNING"  # RUNNING | FINISHED | FAILED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict]] = field(default_factory=dict)  # {name: [{step, value, ts}]}
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    def duration_s(self) -> float:
        t = self.end_time or time.time()
        return t - self.start_time


class ExperimentTracker:
    """
    File-based experiment tracker for RL research.

    Stores all run data as JSON for portability. Can optionally
    forward metrics to MLflow when available.

    Args:
        tracking_dir: Root directory for experiment data
        experiment_name: Logical grouping for related runs
        use_mlflow: Whether to also log to MLflow server
        mlflow_uri: MLflow tracking server URI
    """

    def __init__(
        self,
        tracking_dir: str = "experiments",
        experiment_name: str = "default",
        use_mlflow: bool = False,
        mlflow_uri: str = "http://localhost:5000",
    ):
        self.tracking_dir = Path(tracking_dir) / experiment_name
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.current_run: Optional[Run] = None

        # Optional MLflow integration
        self._mlflow = None
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(experiment_name)
                self._mlflow = mlflow
                logger.info(f"MLflow connected: {mlflow_uri}")
            except ImportError:
                logger.warning("MLflow not installed; falling back to local tracking")

        logger.info(f"ExperimentTracker: {self.tracking_dir}")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Start a new experiment run.

        Args:
            run_name: Human-readable name (auto-generated if None)
            tags: Metadata tags for the run

        Returns:
            run_id: Unique run identifier
        """
        run_id = str(uuid.uuid4())[:8]
        name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_run = Run(
            run_id=run_id,
            experiment_name=self.experiment_name,
            tags={"run_name": name, **(tags or {})},
        )

        if self._mlflow:
            self._mlflow.start_run(run_name=name, tags=tags)

        logger.info(f"Started run: {run_id} ({name})")
        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters (call once before training)."""
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        self.current_run.params.update(params)
        if self._mlflow:
            self._mlflow.log_params(params)

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a scalar metric value.

        Args:
            name: Metric name (e.g., "train/loss", "eval/return")
            value: Metric value
            step: Training step (auto-incremented if None)
        """
        if self.current_run is None:
            raise RuntimeError("No active run.")

        if name not in self.current_run.metrics:
            self.current_run.metrics[name] = []

        entry = {
            "step": step if step is not None else len(self.current_run.metrics[name]),
            "value": float(value),
            "timestamp": time.time(),
        }
        self.current_run.metrics[name].append(entry)

        if self._mlflow:
            self._mlflow.log_metric(name, value, step=entry["step"])

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, path: str) -> None:
        """Register an artifact file path."""
        if self.current_run:
            self.current_run.artifacts.append(path)
        if self._mlflow:
            self._mlflow.log_artifact(path)

    def end_run(self, status: str = "FINISHED") -> None:
        """
        Finalize the current run and persist to disk.

        Args:
            status: Final status (FINISHED | FAILED)
        """
        if self.current_run is None:
            return

        self.current_run.status = status
        self.current_run.end_time = time.time()

        # Persist run data
        run_file = self.tracking_dir / f"{self.current_run.run_id}.json"
        with open(run_file, "w") as f:
            json.dump(asdict(self.current_run), f, indent=2)

        if self._mlflow:
            self._mlflow.end_run()

        logger.info(
            f"Run {self.current_run.run_id} ended: {status} "
            f"({self.current_run.duration_s():.1f}s)"
        )
        self.current_run = None

    def load_run(self, run_id: str) -> Run:
        """Load a previously saved run by ID."""
        run_file = self.tracking_dir / f"{run_id}.json"
        with open(run_file) as f:
            data = json.load(f)
        return Run(**data)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return summary of all runs in the experiment."""
        runs = []
        for run_file in sorted(self.tracking_dir.glob("*.json")):
            with open(run_file) as f:
                data = json.load(f)
            summary = {
                "run_id": data["run_id"],
                "status": data["status"],
                "duration_s": (data.get("end_time") or time.time()) - data["start_time"],
                "params": data["params"],
                "tags": data["tags"],
                "final_metrics": {
                    k: v[-1]["value"] for k, v in data["metrics"].items() if v
                },
            }
            runs.append(summary)
        return runs

    def get_best_run(self, metric: str, mode: str = "max") -> Optional[Dict]:
        """
        Return the run with the best value of a given metric.

        Args:
            metric: Metric name to compare
            mode: "max" or "min"

        Returns:
            Best run summary dict
        """
        runs = self.list_runs()
        eligible = [r for r in runs if metric in r["final_metrics"]]
        if not eligible:
            return None
        key = lambda r: r["final_metrics"][metric]
        return max(eligible, key=key) if mode == "max" else min(eligible, key=key)

    def compare_runs(self, metric: str) -> List[Dict]:
        """Return all runs sorted by metric (descending)."""
        runs = self.list_runs()
        eligible = [r for r in runs if metric in r["final_metrics"]]
        return sorted(
            eligible,
            key=lambda r: r["final_metrics"][metric],
            reverse=True,
        )
