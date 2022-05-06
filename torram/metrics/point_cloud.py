import torch
from typing import Tuple

__all__ = ['acceleration',
           'pve',
           'pa_pve']


def acceleration(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """"Point cloud acceleration error i.e. the euclidean distance between the acceleration of the predicted point cloud
    (w.r.t. gt positions at k - 1 and k + 1) and the acceleration of ground-truth point cloud.
    Inspired by: https://github.com/akanazawa/human_dynamics/

    acc_error = 1/(n-2) sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}

    Args:
        x_hat: predicted point positions at time-step k (B, N, 3).
        x: ground-truth point positions at time-steps [k-1, k, k+1] (B, 3, N, 3).
    """
    if x_hat.ndim != 3 or x_hat.shape[-1] != 3:
        raise ValueError(f"Invalid prediction shape, expected (B, N, 3), got {x_hat.shape}")
    if x.ndim != 4 or x.shape[-1] != 3 or x.shape[1] != 3:
        raise ValueError(f"Invalid target shape, expected (B, 3, N, 3), got {x.shape}")
    if len(x_hat) != len(x) or x_hat.shape[-2] != x.shape[-2]:
        raise ValueError(f"Not matching prediction and target, got {x.shape} and {x_hat.shape}")

    accel = x[:, 0] - 2 * x[:, 1] + x[:, 2]
    accel_hat = x[:, 0] - 2 * x_hat + x[:, 2]
    return torch.mean((accel_hat - accel).norm(dim=-1), dim=-1)


def pve(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean per vertex error using the raw point cloud, i.e. without alignments.

    Args:
        x_hat: predicted vertices (B, N, 3).
        x: ground-truth vertices (B, N, 3).
    """
    __check_matching_3d_point_clouds(x_hat, x)
    return torch.mean((x_hat - x).norm(dim=-1), dim=-1)


def pa_pve(x_hat: torch.Tensor, x: torch.Tensor, align_scale: bool = True) -> torch.Tensor:
    """Per Vertex Position Error after Procrustes alignment. Inspired by the mean per joint position error
    from the human body pose estimation field
    https://github.com/akanazawa/human_dynamics/blob/0887f37464c9a079ad7d69c8358cecd0f43c4f2a/src/evaluation/

    Args:
        x_hat: predicted joints (B, N, 3).
        x: ground-truth joints (B, N, 3).
        align_scale: align scale or only rotation and translation?
    """
    __check_matching_3d_point_clouds(x_hat, x)
    batch_size, _, _ = x.shape
    x_hat_similar = torch.zeros_like(x_hat)
    for k in range(batch_size):
        scale, Rot, trans = __align_umeyama(x[k], x_hat[k], align_scale=align_scale)
        x_hat_similar[k] = trans + scale * torch.einsum('ij,bj->bi', Rot, x_hat[k])
    return torch.mean((x_hat_similar - x).norm(dim=-1), dim=-1)


def __align_umeyama(model: torch.Tensor, data: torch.Tensor, align_scale: bool = False
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implementation of the paper: S. Umeyama, Least-Squares Estimation of Transformation Parameters
    Between Two Point Patterns, IEEE Trans. Pattern Anal. Mach. Intell., vol. 13, no. 4, 1991.

    model = s * R * data + t

    Args:
        model: first trajectory (nx3)
        data: second trajectory (nx3)
        align_scale: True is scale parameter should be estimated; default is False

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
    C = 1.0 / n * torch.einsum('ij,il->jl', model_zerocentered, data_zerocentered)
    sigma2 = 1.0 / n * (data_zerocentered ** 2).sum()
    U_svd, D_svd, V_svd = torch.linalg.svd(C)
    D_svd = torch.diag(D_svd)

    S = torch.eye(3, device=model.device, dtype=U_svd.dtype)
    if torch.linalg.det(U_svd) * torch.linalg.det(V_svd) < 0:
        S[2, 2] = -1

    R = torch.einsum('ij,jk,kl->il', U_svd, S, V_svd)
    if align_scale:
        if torch.allclose(sigma2, torch.zeros_like(sigma2)):
            model_rotated = torch.einsum('ij,bj->bi', R, model)
            s = torch.mean(model_rotated / data)
        else:
            s = 1.0 / sigma2 * torch.trace(D_svd * S)
    else:
        s = 1.0
    t = mu_M - s * torch.einsum('ij,j->i', R, mu_D)
    return s, R, t


def __check_matching_3d_point_clouds(x_hat: torch.Tensor, x: torch.Tensor):
    if not x_hat.ndim == x.ndim == 3 or not x.shape[-1] == x_hat.shape[-1] == 3:
        raise ValueError(f"Invalid point cloud shape, expected (B, N, 3), got {x.shape} and {x_hat.shape}")
    if x.shape != x_hat.shape:
        raise ValueError(f"Non matching prediction and target, got {x.shape} and {x_hat.shape}")
