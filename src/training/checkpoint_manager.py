"""
Checkpoint Manager
==================
Enterprise-grade model checkpoint orchestration with:
  - Versioned checkpoint storage
  - Best-model tracking
  - Automatic cleanup of old checkpoints
  - Metadata persistence (hyperparams, metrics)
  - Atomic writes to prevent corruption
"""

import torch
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with versioning and best-model tracking.

    Directory structure:
        base_dir/
          experiment_name/
            run_YYYYMMDD_HHMMSS/
              checkpoint_iter_0010.pt
              checkpoint_iter_0025.pt
              best_model.pt
              metadata.json

    Args:
        base_dir: Root directory for all checkpoints
        experiment_name: Name to group related runs
        max_keep: Maximum number of checkpoints to retain (older deleted)
    """

    def __init__(
        self,
        base_dir: str = "checkpoints",
        experiment_name: str = "default_exp",
        max_keep: int = 5,
    ):
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.max_keep = max_keep

        # Create timestamped run directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / experiment_name / f"run_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints: List[Path] = []
        self.best_metric = -float("inf")
        self.metadata: Dict[str, Any] = {
            "experiment": experiment_name,
            "run_dir": str(self.run_dir),
            "started_at": ts,
            "checkpoints": [],
        }

        logger.info(f"CheckpointManager: {self.run_dir}")

    def save(
        self,
        agent,
        iteration: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save agent checkpoint atomically.

        Uses temp-file + rename pattern to prevent partial writes
        from corrupting checkpoints on failure.

        Args:
            agent: Agent with save_checkpoint method
            iteration: Current training iteration
            metrics: Optional metric dict to store in metadata
            is_best: Whether this is the best checkpoint so far

        Returns:
            path: Saved checkpoint path
        """
        ck_name = f"checkpoint_iter_{iteration:05d}.pt"
        ck_path = self.run_dir / ck_name
        tmp_path = ck_path.with_suffix(".tmp")

        # Save to temp file first
        agent.save_checkpoint(str(tmp_path))
        # Atomic rename
        tmp_path.rename(ck_path)

        self.checkpoints.append(ck_path)
        logger.info(f"Saved checkpoint: {ck_path}")

        # Track best
        if is_best or (
            metrics
            and metrics.get("eval/mean_return", -float("inf")) > self.best_metric
        ):
            best_path = self.run_dir / "best_model.pt"
            shutil.copy2(ck_path, best_path)
            if metrics:
                self.best_metric = metrics.get("eval/mean_return", self.best_metric)
            logger.info(f"New best model: {best_path}")

        # Update metadata
        entry = {
            "iteration": iteration,
            "path": str(ck_path),
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        self.metadata["checkpoints"].append(entry)
        self._save_metadata()

        # Prune old checkpoints
        self._prune()

        return str(ck_path)

    def _prune(self) -> None:
        """Delete oldest checkpoints beyond max_keep limit."""
        # Never delete best_model.pt or the latest checkpoint
        to_prune = self.checkpoints[: -self.max_keep]
        for ck_path in to_prune:
            if ck_path.exists():
                ck_path.unlink()
                logger.debug(f"Pruned: {ck_path}")
        self.checkpoints = self.checkpoints[-self.max_keep :]

    def _save_metadata(self) -> None:
        """Persist experiment metadata as JSON."""
        meta_path = self.run_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def load_best(self, agent, env) -> None:
        """Load the best model checkpoint into agent."""
        best_path = self.run_dir / "best_model.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best model found at {best_path}")
        from ..agents.node2vec_rl import Node2VecRLAgent
        loaded = Node2VecRLAgent.load_checkpoint(str(best_path), env)
        agent.__dict__.update(loaded.__dict__)
        logger.info(f"Loaded best model from {best_path}")

    def load_latest(self, agent, env) -> None:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")
        latest = self.checkpoints[-1]
        from ..agents.node2vec_rl import Node2VecRLAgent
        loaded = Node2VecRLAgent.load_checkpoint(str(latest), env)
        agent.__dict__.update(loaded.__dict__)
        logger.info(f"Loaded latest checkpoint from {latest}")

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """Return all checkpoint metadata entries."""
        return self.metadata["checkpoints"]
