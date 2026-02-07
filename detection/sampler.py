"""
Hard Negative Mining Sampler for balanced training.

Handles extreme class imbalance in anchor-based detection by:
1. Keeping all positive anchors
2. Sampling hard negatives (high-scoring false positives)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class HardNegativeSampler(nn.Module):
    """
    Hard Negative Mining Sampler.

    Balances positive/negative samples by:
    1. Including all positive samples
    2. Sampling negatives from highest-scoring false positives

    Args:
        batch_size: Total number of anchors to sample per image
        positive_fraction: Target fraction of positives in batch
        pool_size_factor: Sample negatives from top (pool_size_factor * batch_size) scoring
    """

    def __init__(
        self,
        batch_size: int = 256,
        positive_fraction: float = 0.25,
        pool_size_factor: int = 10,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.pool_size_factor = pool_size_factor

    @torch.no_grad()
    def forward(
        self,
        labels: torch.Tensor,
        cls_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample positive and negative anchors.

        Args:
            labels: (N,) labels from matcher (1=pos, 0=neg, -1=ignore)
            cls_scores: (N,) or (N, 1) classification scores for hard mining

        Returns:
            pos_mask: (N,) boolean mask for positive samples
            neg_mask: (N,) boolean mask for negative samples
        """
        device = labels.device
        num_anchors = labels.shape[0]

        # Find positive and negative indices
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        # Determine number of samples
        max_pos = int(self.batch_size * self.positive_fraction)
        num_pos_samples = min(num_pos, max_pos)
        num_neg_samples = self.batch_size - num_pos_samples

        # Sample positives (random if more than max)
        if num_pos > num_pos_samples:
            perm = torch.randperm(num_pos, device=device)[:num_pos_samples]
            sampled_pos = pos_indices[perm]
        else:
            sampled_pos = pos_indices

        # Sample negatives (hard mining or random)
        if num_neg > num_neg_samples:
            if cls_scores is not None:
                # Hard negative mining: sample from highest-scoring negatives
                neg_scores = cls_scores[neg_indices]
                if neg_scores.dim() > 1:
                    neg_scores = neg_scores.squeeze(-1)

                # Pool size for sampling
                pool_size = min(num_neg, self.pool_size_factor * num_neg_samples)
                _, top_indices = neg_scores.topk(pool_size, largest=True)

                # Random sample from pool
                perm = torch.randperm(pool_size, device=device)[:num_neg_samples]
                sampled_neg = neg_indices[top_indices[perm]]
            else:
                # Random sampling
                perm = torch.randperm(num_neg, device=device)[:num_neg_samples]
                sampled_neg = neg_indices[perm]
        else:
            sampled_neg = neg_indices

        # Create masks
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)

        pos_mask[sampled_pos] = True
        neg_mask[sampled_neg] = True

        return pos_mask, neg_mask


class BalancedRandomSampler(nn.Module):
    """
    Simple balanced random sampler without hard mining.

    Args:
        batch_size: Total samples per image
        positive_fraction: Target positive fraction
    """

    def __init__(
        self,
        batch_size: int = 256,
        positive_fraction: float = 0.25,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction

    @torch.no_grad()
    def forward(
        self,
        labels: torch.Tensor,
        cls_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = labels.device
        num_anchors = labels.shape[0]

        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        max_pos = int(self.batch_size * self.positive_fraction)
        num_pos_samples = min(num_pos, max_pos)
        num_neg_samples = self.batch_size - num_pos_samples
        num_neg_samples = min(num_neg_samples, num_neg)

        # Random sampling
        if num_pos > num_pos_samples:
            perm = torch.randperm(num_pos, device=device)[:num_pos_samples]
            sampled_pos = pos_indices[perm]
        else:
            sampled_pos = pos_indices

        if num_neg > num_neg_samples:
            perm = torch.randperm(num_neg, device=device)[:num_neg_samples]
            sampled_neg = neg_indices[perm]
        else:
            sampled_neg = neg_indices

        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)

        pos_mask[sampled_pos] = True
        neg_mask[sampled_neg] = True

        return pos_mask, neg_mask


class OHEMSampler(nn.Module):
    """
    Online Hard Example Mining (OHEM) Sampler.

    Selects the hardest examples based on loss values.

    Args:
        batch_size: Total samples per image
        positive_fraction: Target positive fraction
        min_pos_per_image: Minimum positives per image
    """

    def __init__(
        self,
        batch_size: int = 256,
        positive_fraction: float = 0.25,
        min_pos_per_image: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.positive_fraction = positive_fraction
        self.min_pos_per_image = min_pos_per_image

    @torch.no_grad()
    def forward(
        self,
        labels: torch.Tensor,
        losses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample based on loss values.

        Args:
            labels: (N,) labels (1=pos, 0=neg, -1=ignore)
            losses: (N,) loss values for each anchor

        Returns:
            pos_mask, neg_mask: Boolean masks
        """
        device = labels.device
        num_anchors = labels.shape[0]

        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        # Ensure minimum positives
        max_pos = max(
            int(self.batch_size * self.positive_fraction),
            self.min_pos_per_image
        )
        num_pos_samples = min(num_pos, max_pos)
        num_neg_samples = self.batch_size - num_pos_samples
        num_neg_samples = min(num_neg_samples, num_neg)

        # Sort by loss and take highest
        if num_pos > num_pos_samples:
            pos_losses = losses[pos_indices]
            _, sorted_idx = pos_losses.sort(descending=True)
            sampled_pos = pos_indices[sorted_idx[:num_pos_samples]]
        else:
            sampled_pos = pos_indices

        if num_neg > num_neg_samples:
            neg_losses = losses[neg_indices]
            _, sorted_idx = neg_losses.sort(descending=True)
            sampled_neg = neg_indices[sorted_idx[:num_neg_samples]]
        else:
            sampled_neg = neg_indices

        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)

        pos_mask[sampled_pos] = True
        neg_mask[sampled_neg] = True

        return pos_mask, neg_mask


def subsample_labels(
    labels: torch.Tensor,
    batch_size: int = 256,
    positive_fraction: float = 0.25,
) -> torch.Tensor:
    """
    Subsample labels for balanced training.

    Sets non-sampled labels to -1 (ignore).

    Args:
        labels: (N,) labels (1=pos, 0=neg, -1=ignore)
        batch_size: Total samples
        positive_fraction: Target positive fraction

    Returns:
        labels: (N,) subsampled labels
    """
    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (labels == 0).nonzero(as_tuple=True)[0]

    num_pos = len(pos_indices)
    num_neg = len(neg_indices)

    max_pos = int(batch_size * positive_fraction)
    num_pos_samples = min(num_pos, max_pos)
    num_neg_samples = batch_size - num_pos_samples

    # Random sampling
    if num_pos > num_pos_samples:
        perm = torch.randperm(num_pos, device=labels.device)
        disable = pos_indices[perm[num_pos_samples:]]
        labels[disable] = -1

    if num_neg > num_neg_samples:
        perm = torch.randperm(num_neg, device=labels.device)
        disable = neg_indices[perm[num_neg_samples:]]
        labels[disable] = -1

    return labels
