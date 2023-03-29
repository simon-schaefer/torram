import torch
from typing import Optional


__all__ = ['euclidean_distance',
           'geodesic_loss']


def euclidean_distance(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between to 3D vectors, averaged over batch-size (= L2 loss).

    Args:
        x_hat: predicted vector (B, D).
        x: target vector (B, D).
    """
    assert x_hat.ndim == x.ndim == 2  # (B, D)
    assert x_hat.shape == x.shape
    return (x_hat - x).norm(dim=-1)


def geodesic_loss(x_hat: torch.Tensor, x: Optional[torch.Tensor] = None, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic loss function as difference of rotations.
    https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py

    Args:
        x_hat: predicted rotation matrix (B, 3, 3).
        x: target rotation matrix (B, 3, 3). If None, no rotation is assumed (i.e. R = I).
        eps: numeric eps.
    """
    assert x_hat.shape[-1] == x_hat.shape[-2] == 3  # (B, 3, 3)
    assert x is None or x.shape == x_hat.shape
    if x is not None:
        R_diffs = x_hat @ x.transpose(-1, -2)  # x -> inv(x) = x.T
    else:
        R_diffs = x_hat  # no "back"-rotation
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
