from typing import Optional

import torch
from jaxtyping import Float

__all__ = [
    "geodesic_distance",
    "invert_homogeneous_transforms",
]


def geodesic_distance(
    R_hat: Float[torch.Tensor, "B 3 3"],
    R: Optional[Float[torch.Tensor, "B 3 3"]] = None,
    eps: float = 1e-7,
) -> Float[torch.Tensor, "B"]:
    """Geodesic loss function as difference of rotations.
    https://github.com/airalcorn2/pytorch-geodesic-loss/blob/master/geodesic_loss.py

    @param x_hat: predicted rotation matrix (B, 3, 3).
    @param x: target rotation matrix (B, 3, 3). If None, no rotation is assumed (i.e. R = I).
    @param eps: numeric eps.
    """
    if R is not None:
        R_diffs = R_hat @ R.transpose(-1, -2)  # x -> inv(x) = x.T
    else:
        R_diffs = R_hat  # no "back"-rotation
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))


def invert_homogeneous_transforms(
    T: Float[torch.Tensor, "... 4 4"]
) -> Float[torch.Tensor, "... 4 4"]:
    """Invert homogeneous transformation matrices.

    @param T: homogeneous transformation matrices (..., 4, 4).
    @returns inverted homogeneous transformation matrices (..., 4, 4).
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:]

    T_inv = torch.eye(4, device=T.device, dtype=T.dtype).repeat(*T.shape[:-2], 1, 1).clone()
    R_inv = R.transpose(-1, -2)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3:] = -R_inv @ t

    return T_inv
