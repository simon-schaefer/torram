import torch
from typing import Optional


__all__ = ['euclidean_distance',
           'geodesic_loss',
           'l1_quaternion']


def euclidean_distance(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between to 3D vectors, averaged over batch-size (= L2 loss).

    Args:
        x_hat: predicted vector (B, D).
        x: target vector (B, D).
    """
    if not x_hat.ndim == x.ndim == 2:
        raise ValueError(f"Invalid translation shape, expected (B, D), got {x.shape} and {x_hat.shape}")
    if x_hat.shape != x.shape:
        raise ValueError(f"Non matching prediction and target, got {x.shape} and {x_hat.shape}")
    return (x_hat - x).norm(dim=-1)


def geodesic_loss(x_hat: torch.Tensor, x: Optional[torch.Tensor] = None, eps: float = 1e-7) -> torch.Tensor:
    """Geodesic loss function as difference of rotations.
    https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py

    Args:
        x_hat: predicted rotation matrix (B, 3, 3).
        x: target rotation matrix (B, 3, 3). If None, no rotation is assumed (i.e. R = I).
        eps: numeric eps.
    """
    if not x_hat.shape[-1] == x_hat.shape[-2] == 3:
        raise ValueError(f"Invalid rotation matrix shape, expected (B, 3, 3), got {x_hat.shape}")
    if x is not None:
        if x.shape != x_hat.shape:
            raise ValueError(f"Non matching prediction and target, got {x.shape} and {x_hat.shape}")
        R_diffs = x_hat @ x.transpose(-1, -2)  # x -> inv(x) = x.T
    else:
        R_diffs = x_hat  # no "back"-rotation
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))


def l1_quaternion(x_hat: torch.Tensor, x: torch.Tensor):
    """L1-like quaternion loss function, adapted from:
    https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f

    Args:
        x_hat: predicted rotation in quaternion (B, N, 4).
        x: target rotation in quaternion (B, N, 4).
    """
    if not x_hat.ndim == x.ndim == 3 or not x_hat.shape[-1] == x.shape[-1] == 4:
        raise ValueError(f"Invalid quaternion shape, expected (B, N, 4), got {x.shape} and {x_hat.shape}")
    if x.shape != x_hat.shape:
        raise ValueError(f"Non matching prediction and target, got {x.shape} and {x_hat.shape}")

    x_hat_normed = x_hat / torch.norm(x_hat, dim=-1, keepdim=True)
    l1_plus = torch.norm(x + x_hat_normed, p=1, dim=-1)
    l1_minus = torch.norm(x - x_hat_normed, p=1, dim=-1)
    loss = torch.min(l1_minus, l1_plus)
    return torch.mean(loss, dim=-1)
