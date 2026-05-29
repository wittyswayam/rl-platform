"""
InferNet: Reward Prediction Network
=====================================
Maps graph embeddings to immediate reward predictions, serving as an auxiliary
signal that helps credit assignment across delayed reward trajectories.

Architecture:
    Input (embed_dim) → Linear → Tanh → Linear → Scalar reward prediction

The dual-loss formulation (pointwise + cumulative) ensures the network learns
both fine-grained step-level accuracy and episode-level reward totals.
"""

import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class InferNet(nn.Module):
    """
    Reward inference network for delayed reward credit assignment.

    Uses node embeddings from Node2Vec to predict immediate rewards,
    providing a dense auxiliary learning signal that bridges the gap
    between sparse environmental rewards.

    Args:
        input_dim: Dimension of input node embeddings (must match Node2Vec)
        hidden_dim: Hidden layer width
        output_dim: Output dimension (1 for scalar reward prediction)
        aux_weight: Weight for auxiliary pointwise loss term (default 0.5)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 1,
        aux_weight: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aux_weight = aux_weight

        # Two-layer MLP with tanh activation
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Initialize weights for stable reward prediction
        self._init_weights()
        self.mse = nn.MSELoss(reduction="sum")

        logger.info(
            f"InferNet initialized: {input_dim} → {hidden_dim} → {output_dim}"
        )

    def _init_weights(self) -> None:
        """Xavier initialization for stable gradient flow."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict immediate reward from node embeddings.

        Args:
            embeddings: Node embedding tensor of shape (N, embed_dim)

        Returns:
            predictions: Reward predictions of shape (N, 1)
        """
        return self.network(embeddings)

    def compute_loss(
        self,
        predicted_rewards: torch.Tensor,
        actual_rewards: torch.Tensor,
        walk_length: int,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute dual-objective InferNet loss.

        The loss combines two objectives:
        1. L_main (cumulative): Episode-level total reward accuracy
        2. L_aux (pointwise): Step-level immediate reward accuracy

        Combined: L = L_main + aux_weight * L_aux

        This ensures the network learns to predict both the magnitude of
        individual rewards and the aggregate return of trajectories.

        Args:
            predicted_rewards: Model predictions (B, 1) or (B,)
            actual_rewards: Ground truth rewards (B, 1) or (B,)
            walk_length: Episode length for normalization

        Returns:
            total_loss: Scalar training loss
            metrics: Component loss breakdown
        """
        pred = predicted_rewards.squeeze()
        actual = actual_rewards.squeeze()

        # Pointwise MSE: step-level accuracy
        l_aux = self.mse(pred, actual) / walk_length

        # Cumulative MSE: episode-level accuracy
        pred_sum = pred.sum().unsqueeze(0)
        actual_sum = actual.sum().unsqueeze(0)
        l_main = self.mse(pred_sum, actual_sum) / walk_length

        total_loss = l_main + self.aux_weight * l_aux

        metrics = {
            "loss_main": l_main.item(),
            "loss_aux": l_aux.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def predict_reward_landscape(
        self, all_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict rewards for all nodes (full reward landscape).

        Used during policy improvement to identify high-value regions
        without requiring environment interaction.

        Args:
            all_embeddings: All node embeddings (num_nodes, embed_dim)

        Returns:
            reward_map: Predicted reward for each node (num_nodes,)
        """
        with torch.no_grad():
            return self.forward(all_embeddings).squeeze()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        logger.info(f"InferNet saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "InferNet":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model
