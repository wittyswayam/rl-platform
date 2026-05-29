"""
Benchmark Suite
===============
Standardized benchmarking for RL algorithms on delayed reward tasks.

Provides:
  - Reproducible benchmark configurations
  - Multi-seed statistical evaluation
  - Baseline comparisons (random, oracle)
  - Performance profiling
  - Results formatting for publication
"""

import time
import json
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..envs.graph_nav_env import GraphNavEnv, EnvConfig
from ..agents.node2vec_rl import Node2VecRLAgent, AgentConfig
from ..training.trainer import Trainer, TrainerConfig
from .evaluator import PolicyEvaluator

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    name: str = "default_benchmark"
    seeds: List[int] = None  # [0, 1, 2, 3, 4]
    n_eval_episodes: int = 200
    num_training_iterations: int = 50
    grid_sizes: List[int] = None  # [4, 8, 16]
    coin_densities: List[float] = None  # [0.03, 0.05, 0.10]

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4]
        if self.grid_sizes is None:
            self.grid_sizes = [4, 8, 16]
        if self.coin_densities is None:
            self.coin_densities = [0.03, 0.05, 0.10]


class BenchmarkSuite:
    """
    Standardized benchmark suite for delayed reward RL.

    Runs agents across multiple configurations (grid sizes, reward
    densities, random seeds) and produces publication-ready statistics.

    Args:
        config: Benchmark configuration
        output_dir: Directory for results
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        output_dir: str = "benchmark_results",
    ):
        self.config = config or BenchmarkConfig()
        self.output_dir = output_dir
        import os; os.makedirs(output_dir, exist_ok=True)
        self.results: Dict[str, Any] = {}
        logger.info(f"BenchmarkSuite: {len(self.config.seeds)} seeds, {self.config.name}")

    def _make_coin_nodes(self, num_nodes: int, density: float) -> set:
        """Generate coin node set for given density."""
        n_coins = max(1, int(num_nodes * density))
        return set(np.random.choice(num_nodes, n_coins, replace=False).tolist())

    def run_single(
        self,
        grid_size: int,
        coin_density: float,
        seed: int,
    ) -> Dict[str, Any]:
        """
        Run a single benchmark configuration.

        Args:
            grid_size: Grid dimension (G×G)
            coin_density: Fraction of nodes with coins
            seed: Random seed

        Returns:
            result: Training + evaluation metrics
        """
        np.random.seed(seed)
        num_nodes = grid_size * grid_size
        coin_nodes = self._make_coin_nodes(num_nodes, coin_density)

        env_config = EnvConfig(
            grid_size=grid_size,
            coin_nodes=coin_nodes,
            max_steps=grid_size * grid_size * 2,
        )
        env = GraphNavEnv(config=env_config)

        agent_config = AgentConfig(
            embed_dim=min(512, num_nodes * 8),
            num_iter=self.config.num_training_iterations,
        )
        agent = Node2VecRLAgent(env, agent_config)

        trainer_config = TrainerConfig(
            experiment_name=f"bench_{grid_size}g_{coin_density:.2f}d_s{seed}",
            seed=seed,
            num_iterations=self.config.num_training_iterations,
            output_dir=self.output_dir,
            checkpoint_dir=f"{self.output_dir}/checkpoints",
        )

        t0 = time.time()
        trainer = Trainer(agent, env, trainer_config)
        train_result = trainer.train()
        train_time = time.time() - t0

        evaluator = PolicyEvaluator(env, n_eval_episodes=self.config.n_eval_episodes, seed=seed)
        eval_result = evaluator.evaluate_policy(agent.policy)
        random_result = evaluator.evaluate_random_baseline()

        return {
            "grid_size": grid_size,
            "coin_density": coin_density,
            "seed": seed,
            "train_time_s": train_time,
            "final_return": eval_result["mean_return"],
            "return_std": eval_result["std_return"],
            "coin_rate": eval_result["coin_discovery_rate"],
            "random_return": random_result["mean_return"],
            "improvement_over_random": (
                eval_result["mean_return"] - random_result["mean_return"]
            ),
            "training_history": [
                m.get("train/mean_return", 0)
                for m in train_result["history"]
            ],
        }

    def run_all(self) -> Dict[str, Any]:
        """
        Run full benchmark suite across all configurations and seeds.

        Returns:
            all_results: Aggregated results with statistics
        """
        all_results = []
        total_runs = (
            len(self.config.grid_sizes)
            * len(self.config.coin_densities)
            * len(self.config.seeds)
        )
        run_idx = 0

        for grid_size in self.config.grid_sizes:
            for density in self.config.coin_densities:
                for seed in self.config.seeds:
                    run_idx += 1
                    logger.info(
                        f"[{run_idx}/{total_runs}] Grid={grid_size}, "
                        f"Density={density:.2f}, Seed={seed}"
                    )
                    try:
                        result = self.run_single(grid_size, density, seed)
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Run failed: {e}")
                        all_results.append({"error": str(e), "grid_size": grid_size})

        # Aggregate statistics
        aggregated = self._aggregate(all_results)
        self.results = {"runs": all_results, "aggregated": aggregated}

        results_path = f"{self.output_dir}/{self.config.name}_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Benchmark complete. Results: {results_path}")
        return self.results

    def _aggregate(self, results: List[Dict]) -> Dict:
        """Compute mean/std across seeds for each (grid_size, density) pair."""
        from itertools import groupby
        aggregated = {}

        for grid_size in self.config.grid_sizes:
            for density in self.config.coin_densities:
                runs = [
                    r for r in results
                    if r.get("grid_size") == grid_size
                    and r.get("coin_density") == density
                    and "error" not in r
                ]
                if not runs:
                    continue
                key = f"g{grid_size}_d{density:.2f}"
                returns = [r["final_return"] for r in runs]
                aggregated[key] = {
                    "grid_size": grid_size,
                    "coin_density": density,
                    "mean_return": float(np.mean(returns)),
                    "std_return": float(np.std(returns)),
                    "n_seeds": len(runs),
                    "mean_train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
                    "mean_improvement": float(np.mean([r["improvement_over_random"] for r in runs])),
                }
        return aggregated

    def print_leaderboard(self) -> None:
        """Print benchmark results as formatted leaderboard."""
        if not self.results:
            print("No results yet. Run run_all() first.")
            return

        print("\n" + "="*70)
        print(f"{'BENCHMARK RESULTS':^70}")
        print("="*70)
        print(f"{'Config':<25} {'Mean Return':>12} {'Std':>8} {'vs Random':>12}")
        print("-"*70)

        for key, stats in sorted(
            self.results["aggregated"].items(),
            key=lambda x: x[1]["mean_return"],
            reverse=True,
        ):
            print(
                f"{key:<25} "
                f"{stats['mean_return']:>12.3f} "
                f"{stats['std_return']:>8.3f} "
                f"{stats['mean_improvement']:>+12.3f}"
            )
        print("="*70)
