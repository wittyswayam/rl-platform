"""
Hyperparameter Search
======================
Grid search, random search, and Bayesian optimization for RL hyperparameters.

Supports:
  - Grid search (exhaustive, small parameter spaces)
  - Random search (Bergstra & Bengio 2012)
  - TPE via Optuna (if installed)
  - Parallel execution across parameter configurations

Reference:
  Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (JMLR 2012)
"""

import itertools
import numpy as np
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single hyperparameter configuration result."""
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, float]
    duration_s: float
    trial_id: int


class HyperparameterSearch:
    """
    Multi-strategy hyperparameter optimization for RL agents.

    Runs training with different hyperparameter configurations,
    collecting results for each. Reports the best configuration.

    Args:
        objective_fn: Function (params_dict) → score (higher is better)
        param_space: Parameter space definition
        n_trials: Max trials for random/Bayesian search
        output_dir: Directory for search results
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict], Tuple[float, Dict]],
        param_space: Dict[str, Any],
        n_trials: int = 20,
        output_dir: str = "hparam_search",
    ):
        self.objective_fn = objective_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.output_dir = output_dir
        import os; os.makedirs(output_dir, exist_ok=True)

        self.results: List[SearchResult] = []
        logger.info(f"HyperparameterSearch: {n_trials} trials")

    def _sample_random(self, space: Dict) -> Dict:
        """Sample one random configuration from the parameter space."""
        params = {}
        for key, spec in space.items():
            if isinstance(spec, list):
                params[key] = np.random.choice(spec)
            elif isinstance(spec, dict):
                low, high = spec["low"], spec["high"]
                if spec.get("log", False):
                    params[key] = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
                elif isinstance(low, float) or isinstance(high, float):
                    params[key] = float(np.random.uniform(low, high))
                else:
                    params[key] = int(np.random.randint(low, high + 1))
            else:
                params[key] = spec
        return params

    def grid_search(self) -> SearchResult:
        """
        Exhaustive grid search over all parameter combinations.

        Warning: Exponential in number of parameters.
        Best for ≤4 parameters with small option sets.

        Returns:
            Best SearchResult found
        """
        # Only works with list-type specs
        grid_params = {
            k: v for k, v in self.param_space.items() if isinstance(v, list)
        }
        fixed_params = {
            k: v for k, v in self.param_space.items() if not isinstance(v, list)
        }

        keys = list(grid_params.keys())
        values = [grid_params[k] for k in keys]
        combinations = list(itertools.product(*values))

        logger.info(f"Grid search: {len(combinations)} configurations")

        for trial_id, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            params.update(fixed_params)
            self._run_trial(trial_id, params)

        return self.best_result

    def random_search(self) -> SearchResult:
        """
        Random hyperparameter search.

        Typically more efficient than grid search for high-dimensional spaces,
        as each sample exercises all parameters independently.

        Returns:
            Best SearchResult found
        """
        logger.info(f"Random search: {self.n_trials} trials")

        for trial_id in range(self.n_trials):
            params = self._sample_random(self.param_space)
            self._run_trial(trial_id, params)

        return self.best_result

    def optuna_search(self, direction: str = "maximize") -> SearchResult:
        """
        Bayesian optimization via Optuna (TPE sampler).

        Significantly more sample-efficient than random search
        by modeling the objective function.

        Args:
            direction: "maximize" or "minimize"

        Returns:
            Best SearchResult found
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not installed. Falling back to random search.")
            return self.random_search()

        def objective(trial):
            params = {}
            for key, spec in self.param_space.items():
                if isinstance(spec, list):
                    params[key] = trial.suggest_categorical(key, spec)
                elif isinstance(spec, dict):
                    low, high = spec["low"], spec["high"]
                    if isinstance(low, float):
                        if spec.get("log", False):
                            params[key] = trial.suggest_float(key, low, high, log=True)
                        else:
                            params[key] = trial.suggest_float(key, low, high)
                    else:
                        params[key] = trial.suggest_int(key, low, high)
                else:
                    params[key] = spec

            score, metrics = self.objective_fn(params)
            trial_id = trial.number
            self.results.append(SearchResult(
                params=params, score=score, metrics=metrics,
                duration_s=0.0, trial_id=trial_id
            ))
            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=self.n_trials)

        logger.info(f"Best params: {study.best_params}, score={study.best_value:.4f}")
        return self.best_result

    def _run_trial(self, trial_id: int, params: Dict) -> SearchResult:
        """Execute one trial and record results."""
        logger.info(f"Trial {trial_id}: {params}")
        t0 = time.time()

        try:
            score, metrics = self.objective_fn(params)
        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")
            score = float("-inf")
            metrics = {"error": str(e)}

        duration = time.time() - t0
        result = SearchResult(
            params=params, score=score, metrics=metrics,
            duration_s=duration, trial_id=trial_id
        )
        self.results.append(result)
        logger.info(f"Trial {trial_id}: score={score:.4f}, time={duration:.1f}s")
        return result

    @property
    def best_result(self) -> Optional[SearchResult]:
        """Return the result with highest score."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.score)

    def save_results(self) -> str:
        """Save all trial results to JSON."""
        data = [
            {
                "trial_id": r.trial_id,
                "score": r.score,
                "params": r.params,
                "metrics": r.metrics,
                "duration_s": r.duration_s,
            }
            for r in self.results
        ]
        path = f"{self.output_dir}/search_results.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Search results saved: {path}")
        return path

    def print_summary(self) -> None:
        """Print ranked summary of all trials."""
        if not self.results:
            print("No results yet.")
            return

        sorted_results = sorted(self.results, key=lambda r: r.score, reverse=True)
        print("\n" + "="*60)
        print(f"{'HYPERPARAMETER SEARCH RESULTS':^60}")
        print("="*60)
        for rank, r in enumerate(sorted_results[:10], 1):
            print(f"\nRank {rank} | Score: {r.score:.4f} | Trial {r.trial_id}")
            for k, v in r.params.items():
                print(f"  {k}: {v}")
        print("="*60)
