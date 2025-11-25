from typing import Optional

import torch
from jaxtyping import Float
from kornia.geometry.conversions import axis_angle_to_rotation_matrix

__all__ = [
    "geodesic_distance",
    "invert_homogeneous_transforms",
    "rotation_matrix_from_two_vectors",
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


def rotation_matrix_from_two_vectors(
    v1: Float[torch.Tensor, "B 3"],
    v2: Float[torch.Tensor, "B 3"],
) -> Float[torch.Tensor, "B 3 3"]:
    """Compute rotation matrix that rotates v1 to v2.

    @param v1: source vectors (B, 3).
    @param v2: target vectors (B, 3).
    @returns rotation matrices (B, 3, 3).
    """
    v1 = torch.nn.functional.normalize(v1)
    v2 = torch.nn.functional.normalize(v2)
    cross = torch.linalg.cross(v1, v2)
    dot = torch.sum(v1 * v2, dim=-1)

    if torch.any(dot < -0.999999):
        mask = dot < -0.999999

        axis = torch.zeros_like(cross)
        angle = torch.zeros_like(dot)

        # 180° rotation: choose an arbitrary perpendicular axis.
        x = torch.zeros_like(v1[mask])
        x[:, 0] = 1.0
        axis[mask] = torch.nn.functional.normalize(torch.cross(v1[mask], x))
        angle[mask] = torch.pi

        if torch.any(torch.linalg.norm(axis[mask], dim=-1) < 1e-6):
            submask = torch.linalg.norm(axis[mask], dim=-1) < 1e-6
            v1_ = v1[mask][submask]
            y = torch.zeros_like(v1_)
            y[:, 1] = 1.0
            axis[mask][submask] = torch.nn.functional.normalize(torch.cross(v1_, y))

        # General case.
        axis[~mask] = torch.nn.functional.normalize(cross[~mask])
        angle[~mask] = torch.arccos(torch.clamp(dot[~mask], -1.0, 1.0))

    else:
        axis = torch.nn.functional.normalize(cross)
        angle = torch.arccos(torch.clamp(dot, -1.0, 1.0))

    return axis_angle_to_rotation_matrix(axis * angle[:, None])
