"""
Curriculum Learning
====================
Progressive difficulty scheduling for RL training on graph navigation.

Implements several curriculum strategies:
  - Linear difficulty progression
  - Performance-gated advancement
  - Automatic curriculum via success rate tracking
  - Multi-stage curriculum with configurable transitions

The curriculum principle: start with easy tasks (small grids, dense rewards)
and progressively increase difficulty as the agent improves.

Reference:
  Bengio et al., "Curriculum Learning" (ICML 2009)
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ..envs.graph_nav_env import GraphNavEnv, EnvConfig

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    LINEAR = "linear"
    PERFORMANCE_GATED = "performance_gated"
    EXPONENTIAL = "exponential"


@dataclass
class Stage:
    """Single curriculum stage definition."""
    name: str
    grid_size: int
    coin_density: float
    max_steps_multiplier: float = 2.0
    success_threshold: float = 0.5  # Required success rate to advance
    min_episodes: int = 50


class CurriculumScheduler:
    """
    Progressive difficulty scheduler for RL training.

    Automatically advances through curriculum stages based on:
      - Fixed schedules (linear/exponential)
      - Performance thresholds (coin discovery rate)

    Args:
        stages: Ordered list of curriculum stages
        strategy: Advancement strategy
        window_size: Episodes to measure success rate over
    """

    def __init__(
        self,
        stages: Optional[List[Stage]] = None,
        strategy: CurriculumStrategy = CurriculumStrategy.PERFORMANCE_GATED,
        window_size: int = 50,
    ):
        if stages is None:
            stages = self._default_stages()

        self.stages = stages
        self.strategy = strategy
        self.window_size = window_size

        self.current_stage_idx = 0
        self.episodes_at_stage = 0
        self.success_history: List[float] = []
        self.stage_transitions: List[Dict] = []

        logger.info(
            f"CurriculumScheduler: {len(stages)} stages, "
            f"strategy={strategy.value}"
        )

    def _default_stages(self) -> List[Stage]:
        """Default 4-stage curriculum from easy to hard."""
        return [
            Stage(name="tiny_dense", grid_size=4, coin_density=0.20,
                  success_threshold=0.70, min_episodes=30),
            Stage(name="small_medium", grid_size=4, coin_density=0.08,
                  success_threshold=0.60, min_episodes=50),
            Stage(name="medium_sparse", grid_size=8, coin_density=0.05,
                  success_threshold=0.50, min_episodes=75),
            Stage(name="full_task", grid_size=8, coin_density=0.03,
                  success_threshold=0.40, min_episodes=100),
        ]

    @property
    def current_stage(self) -> Stage:
        """Return current curriculum stage."""
        return self.stages[min(self.current_stage_idx, len(self.stages) - 1)]

    @property
    def is_complete(self) -> bool:
        """True if all curriculum stages are complete."""
        return self.current_stage_idx >= len(self.stages)

    def make_env(self) -> GraphNavEnv:
        """
        Create environment for current curriculum stage.

        Returns:
            env: GraphNavEnv configured for current difficulty
        """
        stage = self.current_stage
        num_nodes = stage.grid_size ** 2
        n_coins = max(1, int(num_nodes * stage.coin_density))
        coin_nodes = set(
            np.random.choice(num_nodes, n_coins, replace=False).tolist()
        )
        config = EnvConfig(
            grid_size=stage.grid_size,
            coin_nodes=coin_nodes,
            max_steps=int(num_nodes * stage.max_steps_multiplier),
        )
        return GraphNavEnv(config=config)

    def record_episode(self, success: bool) -> None:
        """
        Record episode outcome and check for stage advancement.

        Args:
            success: Whether the episode collected at least one coin
        """
        self.success_history.append(float(success))
        self.episodes_at_stage += 1

        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)

        if self.strategy == CurriculumStrategy.PERFORMANCE_GATED:
            self._check_advance_performance()

    def _check_advance_performance(self) -> None:
        """Advance if success rate exceeds threshold for minimum episodes."""
        stage = self.current_stage

        if self.episodes_at_stage < stage.min_episodes:
            return

        if len(self.success_history) < 10:
            return

        success_rate = np.mean(self.success_history)

        if success_rate >= stage.success_threshold:
            self._advance()

    def _advance(self) -> None:
        """Move to next curriculum stage."""
        old_stage = self.current_stage.name
        self.stage_transitions.append({
            "from_stage": old_stage,
            "to_stage_idx": self.current_stage_idx + 1,
            "episodes_taken": self.episodes_at_stage,
            "final_success_rate": np.mean(self.success_history),
        })

        self.current_stage_idx += 1
        self.episodes_at_stage = 0
        self.success_history = []

        if not self.is_complete:
            logger.info(
                f"Curriculum advanced: {old_stage} → {self.current_stage.name}"
            )
        else:
            logger.info("Curriculum complete! All stages passed.")

    def linear_advance(self, total_iterations: int, current_iteration: int) -> None:
        """
        Advance based on linear schedule (ignore performance).

        Args:
            total_iterations: Total training iterations
            current_iteration: Current training iteration
        """
        stage_len = total_iterations // len(self.stages)
        target_stage = min(
            current_iteration // stage_len, len(self.stages) - 1
        )
        if target_stage > self.current_stage_idx:
            self.current_stage_idx = target_stage
            logger.info(f"Linear advance to stage: {self.current_stage.name}")

    def get_progress(self) -> Dict[str, Any]:
        """Return curriculum progress summary."""
        return {
            "current_stage": self.current_stage.name,
            "stage_idx": self.current_stage_idx,
            "total_stages": len(self.stages),
            "episodes_at_stage": self.episodes_at_stage,
            "recent_success_rate": float(np.mean(self.success_history)) if self.success_history else 0.0,
            "is_complete": self.is_complete,
            "transitions": self.stage_transitions,
        }
