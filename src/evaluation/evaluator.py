"""
Evaluation Pipeline
====================
Comprehensive policy evaluation with statistical analysis.

Provides:
  - Multi-episode Monte Carlo evaluation
  - Confidence interval computation
  - Comparison against baselines (random, optimal)
  - Per-state value estimation
  - Visualization-ready output
"""

import numpy as np
import torch
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats as scipy_stats

from ..envs.graph_nav_env import GraphNavEnv

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Statistical policy evaluation for graph navigation agents.

    Evaluates policies via Monte Carlo rollouts and computes:
      - Mean/std episode return
      - Confidence intervals (bootstrap or t-test)
      - Coin discovery rate
      - Comparison against random and oracle baselines

    Args:
        env: Graph navigation environment
        n_eval_episodes: Episodes per evaluation
        confidence_level: CI confidence (default 95%)
        seed: Evaluation seed for reproducibility
    """

    def __init__(
        self,
        env: GraphNavEnv,
        n_eval_episodes: int = 100,
        confidence_level: float = 0.95,
        seed: int = 0,
    ):
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.confidence_level = confidence_level
        self.seed = seed
        np.random.seed(seed)
        logger.info(
            f"PolicyEvaluator: {n_eval_episodes} episodes, "
            f"CI={confidence_level:.0%}"
        )

    def evaluate_policy(
        self,
        policy: Dict[int, int],
        n_episodes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a deterministic policy via Monte Carlo rollouts.

        Args:
            policy: {state: action} dict
            n_episodes: Override episode count

        Returns:
            results: Statistical evaluation results
        """
        n = n_episodes or self.n_eval_episodes
        returns, coin_counts, lengths = [], [], []

        for ep in range(n):
            start = np.random.randint(0, self.env.num_nodes)
            episode = self.env.sample_episode(
                policy=policy, start_node=start
            )
            ep_return = sum(episode["rewards"])
            coins = sum(1 for r in episode["rewards"] if r > 0)
            returns.append(ep_return)
            coin_counts.append(coins)
            lengths.append(len(episode["actions"]))

        returns_arr = np.array(returns)
        ci = self._bootstrap_ci(returns_arr, self.confidence_level)

        return {
            "mean_return": float(np.mean(returns_arr)),
            "std_return": float(np.std(returns_arr)),
            "min_return": float(np.min(returns_arr)),
            "max_return": float(np.max(returns_arr)),
            "median_return": float(np.median(returns_arr)),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "mean_coins": float(np.mean(coin_counts)),
            "coin_discovery_rate": float(np.mean([c > 0 for c in coin_counts])),
            "mean_episode_length": float(np.mean(lengths)),
            "n_episodes": n,
            "returns": returns,
        }

    def _bootstrap_ci(
        self, values: np.ndarray, confidence: float, n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for the mean.

        Args:
            values: Sample array
            confidence: Confidence level [0, 1]
            n_bootstrap: Bootstrap resamples

        Returns:
            (lower, upper) confidence interval bounds
        """
        means = [np.mean(np.random.choice(values, len(values))) for _ in range(n_bootstrap)]
        alpha = 1 - confidence
        lower = np.percentile(means, 100 * alpha / 2)
        upper = np.percentile(means, 100 * (1 - alpha / 2))
        return lower, upper

    def evaluate_random_baseline(self) -> Dict[str, Any]:
        """Evaluate uniform random policy as lower-bound baseline."""
        random_policy = {
            n: np.random.randint(0, self.env.num_actions)
            for n in range(self.env.num_nodes)
        }
        result = self.evaluate_policy(random_policy)
        result["policy_type"] = "random"
        return result

    def compare_policies(
        self,
        policies: Dict[str, Dict[int, int]],
    ) -> Dict[str, Any]:
        """
        Compare multiple named policies statistically.

        Args:
            policies: {policy_name: {state: action}} dict

        Returns:
            comparison: Per-policy results + significance tests
        """
        results = {}
        for name, policy in policies.items():
            logger.info(f"Evaluating policy: {name}")
            results[name] = self.evaluate_policy(policy)

        # Pairwise significance tests
        policy_names = list(results.keys())
        significance = {}
        for i in range(len(policy_names)):
            for j in range(i + 1, len(policy_names)):
                n1, n2 = policy_names[i], policy_names[j]
                r1 = np.array(results[n1]["returns"])
                r2 = np.array(results[n2]["returns"])
                stat, pval = scipy_stats.mannwhitneyu(r1, r2, alternative="two-sided")
                key = f"{n1}_vs_{n2}"
                significance[key] = {
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant": pval < (1 - self.confidence_level),
                }

        return {"policies": results, "significance_tests": significance}

    def compute_value_landscape(
        self,
        policy: Dict[int, int],
        gamma: float = 0.99,
        n_rollouts: int = 50,
    ) -> np.ndarray:
        """
        Estimate V(s) for all states via Monte Carlo rollouts.

        Args:
            policy: Policy to evaluate
            gamma: Discount factor
            n_rollouts: Rollouts per starting state

        Returns:
            values: V(s) array of shape (num_nodes,)
        """
        values = np.zeros(self.env.num_nodes)

        for start_node in range(self.env.num_nodes):
            rollout_returns = []
            for _ in range(n_rollouts):
                episode = self.env.sample_episode(policy=policy, start_node=start_node)
                rewards = episode["rewards"]
                # Discounted return
                G = sum(gamma ** t * r for t, r in enumerate(rewards))
                rollout_returns.append(G)
            values[start_node] = np.mean(rollout_returns)

        return values

    def save_results(self, results: Dict, path: str) -> None:
        """Persist evaluation results to JSON."""
        # Make JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(make_serializable(results), f, indent=2)
        logger.info(f"Evaluation results saved to {path}")
