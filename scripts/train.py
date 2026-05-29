#!/usr/bin/env python3
"""
Training Entry Point
====================
CLI script for launching RL training runs.

Usage:
    python scripts/train.py --help
    python scripts/train.py --num_iterations 100 --grid_size 8
    python scripts/train.py --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.graph_nav_env import GraphNavEnv, EnvConfig
from src.agents.node2vec_rl import Node2VecRLAgent, AgentConfig
from src.training.trainer import Trainer, TrainerConfig
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.telemetry import TelemetryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RL Agent")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="default_run")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--lr_node2vec", type=float, default=0.1)
    parser.add_argument("--lr_infernet", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--telemetry_port", type=int, default=9090)
    parser.add_argument("--no_checkpoint", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Starting training run: {args.experiment_name}")

    # ── Environment ──
    env_config = EnvConfig(
        grid_size=args.grid_size,
        coin_nodes={10, 30, 50} if args.grid_size == 8 else {3, 7, 12},
        max_steps=args.grid_size ** 2 * 2,
    )
    env = GraphNavEnv(config=env_config)
    logger.info(f"Environment: {args.grid_size}×{args.grid_size} grid")

    # ── Agent ──
    agent_config = AgentConfig(
        embed_dim=args.embed_dim,
        num_iter=args.num_iterations,
        lr_node2vec=args.lr_node2vec,
        lr_infernet=args.lr_infernet,
        device=args.device,
    )
    agent = Node2VecRLAgent(env, agent_config)

    # ── Experiment Tracking ──
    tracker = ExperimentTracker(
        experiment_name=args.experiment_name,
    )
    run_id = tracker.start_run(run_name=f"{args.experiment_name}_s{args.seed}")
    tracker.log_params(vars(args))

    # ── Telemetry ──
    telemetry = TelemetryService(port=args.telemetry_port)
    try:
        telemetry.start_server()
    except Exception as e:
        logger.warning(f"Telemetry server failed to start: {e}")

    # ── Trainer ──
    trainer_config = TrainerConfig(
        experiment_name=args.experiment_name,
        num_iterations=args.num_iterations,
        seed=args.seed,
        checkpoint_dir="checkpoints" if not args.no_checkpoint else "/tmp/rl_ck",
        output_dir="outputs",
    )
    trainer = Trainer(agent, env, trainer_config)

    try:
        results = trainer.train()

        # Log final metrics
        final_eval = results.get("final_eval", {})
        tracker.log_metrics(final_eval)
        tracker.end_run("FINISHED")

        logger.info("Training complete.")
        logger.info(f"Final mean return: {final_eval.get('eval/mean_return', 0):.3f}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        tracker.end_run("FAILED")
        sys.exit(1)
    finally:
        telemetry.stop_server()


if __name__ == "__main__":
    main()
