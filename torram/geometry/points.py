from typing import Tuple

import torch
from jaxtyping import Float

__all__ = ["align_umeyama"]


def align_umeyama(
    model: Float[torch.Tensor, "N 3"],
    data: Float[torch.Tensor, "N 3"],
    align_scale: bool = False,
) -> Tuple[torch.Tensor, Float[torch.Tensor, "3 3"], Float[torch.Tensor, "3 1"]]:
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation of Transformation Parameters
    Between Two Point Patterns, IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Return: [s, R, t]:
        s -> scale factor (scalar)
        R -> rotation matrix (3 x 3)
        t -> translation vector (3 x 1)
    """
    # subtract mean
    mu_M = torch.mean(model, dim=0)
    mu_D = torch.mean(data, dim=0)
    model_zerocentered = model - mu_M
    data_zerocentered = data - mu_D
    n = model.shape[0]

    # correlation
    C = 1.0 / n * torch.einsum("ij,il->jl", model_zerocentered, data_zerocentered)
    sigma2 = 1.0 / n * (data_zerocentered**2).sum()
    U_svd, D_svd, V_svd = torch.linalg.svd(C)
    D_svd = torch.diag(D_svd)

    S = torch.eye(3, device=model.device, dtype=U_svd.dtype)
    if torch.linalg.det(U_svd) * torch.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = torch.einsum("ij,jk,kl->il", U_svd, S, V_svd)
    if align_scale:
        if torch.allclose(sigma2, torch.zeros_like(sigma2)):
            model_rotated = torch.einsum("ij,bj->bi", R, model)
            s = torch.mean(model_rotated / data)
        else:
            s = 1.0 / sigma2 * torch.trace(D_svd * S)
    else:
        s = torch.ones(1, device=model.device, dtype=model.dtype)

    t = mu_M - s * torch.einsum("ij,j->i", R, mu_D)
    return s, R, t


def point2point_distance(
    x_hat: Float[torch.Tensor, "B N 3"],
    x: Float[torch.Tensor, "B N 3"],
    align: bool = False,
    align_scale: bool = True,
) -> Float[torch.Tensor, "B N"]:
    """Point-2-Point distance with pairs determined by same index. Optionally align the point clouds
    using Procrustus alignment.
    """
    if align:
        batch_size, _, _ = x.shape
        x_hat_similar = torch.zeros_like(x_hat)
        for k in range(batch_size):
            scale, Rot, trans = align_umeyama(x[k], x_hat[k], align_scale=align_scale)
            x_hat_similar[k] = trans + scale * torch.einsum("ij,bj->bi", Rot, x_hat[k])
    else:
        x_hat_similar = x_hat

    return (x_hat_similar - x).norm(dim=-1)


def point2point_accel_error(
    x_hat: Float[torch.Tensor, "... T N 3"],
    x: Float[torch.Tensor, "... T N 3"],
) -> Float[torch.Tensor, "... t N"]:
    """Point cloud acceleration error i.e. the distance between the acceleration of `x` and `x_hat`.
    (w.r.t. gt positions at k - 1 and k + 1) and the acceleration of ground-truth point cloud.

    Inspired by:
    https://github.com/akanazawa/human_dynamics/blob/0887f37464c9a079ad7d69c8358cecd0f43c4f2a/src/evaluation/eval_util.py
    & https://github.com/mkocabas/VIBE/blob/851f779407445b75cd1926402f61c931568c6947/lib/utils/eval_utils.py
    """
    accel = point_acceleration(x)
    accel_hat = point_acceleration(x_hat)
    return torch.linalg.norm(accel_hat - accel, dim=-1)


def point_acceleration(x: Float[torch.Tensor, "... T N 3"]) -> Float[torch.Tensor, "... t N 3"]:
    """Point cloud acceleration, i.e. the acceleration of the each individual point in the point cloud."""
    # accel = x[..., :-2, :, :] - 2 * x[..., 1:-1, :, :] + x[..., 2:, :, :]
    velocities = x[..., 1:, :, :] - x[..., :-1, :, :]
    accel = velocities[..., 1:, :, :] - velocities[..., :-1, :, :]
    return accel
