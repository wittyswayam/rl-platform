"""
Node2Vec Graph Embedding Model
===============================
Production-grade implementation of Node2Vec for graph-based RL state representations.
Uses contrastive learning to capture graph topology in dense embedding vectors.

Architecture:
    Embedding Layer (num_nodes × embed_dim)
    → Contrastive Loss (positive + negative pairs)
    → Dense node representations

References:
    - Grover & Leskovec, "node2vec: Scalable Feature Learning for Networks" (KDD 2016)
    - Mikolov et al., "Distributed Representations of Words" (NeurIPS 2013)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Node2VecModel(nn.Module):
    """
    Node2Vec embedding model for graph-structured state spaces.

    Learns continuous node representations by predicting context nodes
    in biased random walks. The embeddings capture both local and global
    graph structure, enabling effective RL state representations.

    Args:
        num_nodes: Number of nodes in the graph
        embedding_dim: Dimension of learned node embeddings
        eps: Numerical stability constant for log computations
    """

    def __init__(
        self,
        num_nodes: int = 64,
        embedding_dim: int = 512,
        eps: float = 1e-15,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.eps = eps

        # Core embedding table: maps node indices to dense vectors
        self.embedding = nn.Embedding(num_nodes, embedding_dim)

        # Initialize with Xavier uniform for stable training
        nn.init.xavier_uniform_(self.embedding.weight)
        logger.info(
            f"Node2VecModel initialized: {num_nodes} nodes × {embedding_dim} dims"
        )

    def forward(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Retrieve embeddings for a batch of node indices.

        Args:
            node_indices: LongTensor of shape (batch_size,)

        Returns:
            embeddings: FloatTensor of shape (batch_size, embedding_dim)
        """
        return self.embedding(node_indices)

    def get_all_embeddings(self) -> torch.Tensor:
        """
        Return embeddings for all nodes.

        Returns:
            all_embeddings: FloatTensor of shape (num_nodes, embedding_dim)
        """
        all_indices = torch.arange(self.num_nodes, dtype=torch.long)
        return self.forward(all_indices)

    def positive_loss(
        self,
        start_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss for positive (co-occurring) node pairs.

        Maximizes dot-product similarity for nodes that appear
        together in random walks (structural proximity).

        Args:
            start_embeddings: Source node embeddings (B, D)
            positive_embeddings: Context node embeddings (B, D)

        Returns:
            loss: Scalar positive pair loss
        """
        # Dot product similarity
        similarity = (start_embeddings * positive_embeddings).sum(dim=-1)
        # Maximize probability of co-occurrence → minimize negative log-sigmoid
        loss = -F.logsigmoid(similarity + self.eps).mean()
        return loss

    def negative_loss(
        self,
        start_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Contrastive loss for negative (non-co-occurring) node pairs.

        Minimizes dot-product similarity for randomly sampled nodes
        that should NOT appear together in walks.

        Args:
            start_embeddings: Source node embeddings (B, D)
            negative_embeddings: Random negative node embeddings (B, D)

        Returns:
            loss: Scalar negative pair loss
        """
        similarity = (start_embeddings * negative_embeddings).sum(dim=-1)
        # Maximize probability of NOT co-occurring
        loss = -F.logsigmoid(-similarity + self.eps).mean()
        return loss

    def compute_loss(
        self,
        start_nodes: torch.Tensor,
        positive_nodes: torch.Tensor,
        negative_nodes: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined Node2Vec contrastive loss.

        Args:
            start_nodes: Walk start node indices (B,)
            positive_nodes: Context node indices (B,)
            negative_nodes: Random negative node indices (B,)

        Returns:
            total_loss: Combined contrastive loss
            metrics: Dictionary with loss component breakdown
        """
        h_start = self.forward(start_nodes)
        h_pos = self.forward(positive_nodes)
        h_neg = self.forward(negative_nodes)

        pos_loss = self.positive_loss(h_start, h_pos)
        neg_loss = self.negative_loss(h_start, h_neg)
        total_loss = pos_loss + neg_loss

        metrics = {
            "positive_loss": pos_loss.item(),
            "negative_loss": neg_loss.item(),
            "total_loss": total_loss.item(),
        }
        return total_loss, metrics

    def cosine_similarity_matrix(self) -> torch.Tensor:
        """
        Compute pairwise cosine similarity between all node embeddings.

        Useful for embedding quality analysis and visualization.

        Returns:
            sim_matrix: FloatTensor of shape (num_nodes, num_nodes)
        """
        embeddings = self.get_all_embeddings()
        normalized = F.normalize(embeddings, dim=-1)
        return torch.mm(normalized, normalized.t())

    def save(self, path: str) -> None:
        """Serialize model weights to disk."""
        torch.save(self.state_dict(), path)
        logger.info(f"Node2VecModel saved to {path}")

    @classmethod
    def load(cls, path: str, **kwargs) -> "Node2VecModel":
        """Load model weights from disk."""
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        logger.info(f"Node2VecModel loaded from {path}")
        return model
