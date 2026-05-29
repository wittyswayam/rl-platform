"""
Core Training Pipeline
=======================
Unified training orchestration for all RL agents on the graph navigation task.

Handles:
  - Training loop management
  - Metric collection and logging
  - Checkpoint scheduling
  - Early stopping
  - Learning rate scheduling
  - Experiment reproducibility
"""

import torch
import numpy as np
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict

from ..agents.node2vec_rl import Node2VecRLAgent, AgentConfig
from ..envs.graph_nav_env import GraphNavEnv, EnvConfig
from .checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Top-level training configuration."""
    # Experiment identity
    experiment_name: str = "node2vec_rl_default"
    run_id: str = "run_001"
    seed: int = 42

    # Training schedule
    num_iterations: int = 100
    eval_every: int = 10
    checkpoint_every: int = 25
    log_every: int = 5

    # Early stopping
    early_stop_patience: int = 20
    early_stop_min_delta: float = 1e-4

    # Paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # Device
    device: str = "cpu"


class Trainer:
    """
    Main training orchestrator for RL agents.

    Provides a clean separation between algorithm logic (in agents)
    and training infrastructure (here), following the single-responsibility
    principle for maintainable research codebases.

    Args:
        agent: RL agent to train
        env: Training environment
        config: Trainer configuration
    """

    def __init__(
        self,
        agent: Node2VecRLAgent,
        env: GraphNavEnv,
        config: Optional[TrainerConfig] = None,
    ):
        self.agent = agent
        self.env = env
        self.config = config or TrainerConfig()

        # Set global seeds for reproducibility
        self._set_seeds(self.config.seed)

        # Infrastructure
        self.checkpoint_manager = CheckpointManager(
            base_dir=self.config.checkpoint_dir,
            experiment_name=self.config.experiment_name,
        )

        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        self.best_return = -np.inf
        self.patience_counter = 0

        # Paths
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Trainer initialized: experiment={self.config.experiment_name}, "
            f"run={self.config.run_id}"
        )

    def _set_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.debug(f"Seeds set to {seed}")

    def _evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate current policy over multiple episodes.

        Args:
            n_episodes: Number of evaluation episodes

        Returns:
            eval_metrics: Dict with mean/std return, coin rate, etc.
        """
        returns = []
        coin_counts = []
        step_counts = []

        for _ in range(n_episodes):
            episode = self.env.sample_episode(
                policy=self.agent.policy,
                start_node=None,
            )
            ep_return = sum(episode["rewards"])
            coins = sum(1 for r in episode["rewards"] if r > 0)
            returns.append(ep_return)
            coin_counts.append(coins)
            step_counts.append(len(episode["actions"]))

        return {
            "eval/mean_return": float(np.mean(returns)),
            "eval/std_return": float(np.std(returns)),
            "eval/mean_coins": float(np.mean(coin_counts)),
            "eval/coin_hit_rate": float(np.mean([c > 0 for c in coin_counts])),
            "eval/mean_steps": float(np.mean(step_counts)),
        }

    def _check_early_stop(self, current_return: float) -> bool:
        """
        Check if training has plateaued (early stopping).

        Args:
            current_return: Current evaluation mean return

        Returns:
            True if training should stop
        """
        if current_return > self.best_return + self.config.early_stop_min_delta:
            self.best_return = current_return
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stop_patience

    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop.

        Returns:
            final_metrics: Complete training history and final evaluation
        """
        logger.info(
            f"Starting training: {self.config.num_iterations} iterations, "
            f"seed={self.config.seed}"
        )
        t_start = time.time()

        for iteration in range(1, self.config.num_iterations + 1):

            # --- Core training step ---
            t0 = time.time()
            trajectories = self.agent._collect_trajectories()
            loss_n2v = self.agent._update_node2vec(trajectories)
            loss_inf = self.agent._update_infernet(trajectories)
            self.agent._update_q_values(trajectories)
            self.agent._extract_policy()
            step_time = time.time() - t0

            # Track training metrics
            ep_return = np.mean([sum(ep["rewards"]) for ep in trajectories.values()])
            self.agent.losses_node2vec.append(loss_n2v)
            self.agent.losses_infernet.append(loss_inf)
            self.agent.episode_returns.append(ep_return)
            self.agent.iteration = iteration

            step_metrics = {
                "iteration": iteration,
                "train/loss_node2vec": loss_n2v,
                "train/loss_infernet": loss_inf,
                "train/mean_return": ep_return,
                "train/step_time_s": step_time,
            }

            # --- Evaluation ---
            if iteration % self.config.eval_every == 0:
                eval_metrics = self._evaluate()
                step_metrics.update(eval_metrics)

                # Early stopping check
                if self._check_early_stop(eval_metrics["eval/mean_return"]):
                    logger.info(
                        f"Early stopping at iteration {iteration}: "
                        f"no improvement for {self.config.early_stop_patience} evals"
                    )
                    break

            # --- Checkpointing ---
            if iteration % self.config.checkpoint_every == 0:
                ck_path = self.checkpoint_manager.save(self.agent, iteration)
                logger.info(f"Checkpoint saved: {ck_path}")

            self.metrics_history.append(step_metrics)

            # --- Logging ---
            if iteration % self.config.log_every == 0:
                elapsed = time.time() - t_start
                logger.info(
                    f"[{iteration:4d}/{self.config.num_iterations}] "
                    f"N2V={loss_n2v:.4f} | Inf={loss_inf:.4f} | "
                    f"Ret={ep_return:.3f} | {elapsed:.1f}s"
                )

        # --- Final evaluation & save ---
        final_eval = self._evaluate(n_episodes=50)
        self.agent.save_checkpoint(
            f"{self.config.checkpoint_dir}/{self.config.experiment_name}_final.pt"
        )

        # Save metrics to JSON
        metrics_path = f"{self.config.output_dir}/{self.config.experiment_name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        total_time = time.time() - t_start
        logger.info(
            f"Training complete in {total_time:.1f}s. "
            f"Final eval return: {final_eval['eval/mean_return']:.3f}"
        )

        return {
            "history": self.metrics_history,
            "final_eval": final_eval,
            "total_time_s": total_time,
        }
