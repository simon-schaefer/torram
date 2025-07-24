from typing import Optional

import torch
from jaxtyping import Float


def geodesic_distance(
    R_hat: Float[torch.Tensor, "B 3 3"],
    R: Optional[Float[torch.Tensor, "B 3 3"]] = None,
    eps: float = 1e-7,
) -> Float[torch.Tensor, "B"]:
    """Geodesic loss function as difference of rotations.
    https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py

    Args:
        x_hat: predicted rotation matrix (B, 3, 3).
        x: target rotation matrix (B, 3, 3). If None, no rotation is assumed (i.e. R = I).
        eps: numeric eps.
    """
    if R is not None:
        R_diffs = R_hat @ R.transpose(-1, -2)  # x -> inv(x) = x.T
    else:
        R_diffs = R_hat  # no "back"-rotation
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))
