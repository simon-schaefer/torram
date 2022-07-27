import torch
import torram

from torch.distributions import Distribution, Normal, MultivariateNormal
from typing import Tuple

__all__ = [
    'kalman_update',
    'kalman_update_with_distributions'
]


def kalman_update(x_hat: torch.Tensor, z: torch.Tensor, P_hat: torch.Tensor, R: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kalman update equations for merging predictions from the process model (x) and measurement (z).

    This implementation assumes that the output of the process model (x) and the measurement (z) are describing
    the same quantity, i.e. that the matrix H with x_z = H * z is the identity matrix. Equations:

    state @ t=k:        (mean: x_hat, variance: P_hat)
    measurement @ t=k:  (mean: z, variance: R)

    K = P_{k|k-1} * (P_{k|k-1} + R_k)^(-1)
    x_k = x_{k|k-1} + K * (z_k - x_{k|k-1})
    P_k = (I - K) * P_{k|k-1} * (I - K)^T + K * R_k * K^T

    The Joseph's form of the measurement update is used to avoid loss of symmetry and positive definiteness due to
    numerical errors (following the description of
    https://www.cs.cmu.edu/~motionplanning/papers/sbpx_%7Bk%7Ck-1%7D_papers/kalman/kleeman_understanding_kalman.pdf).

    Args:
        x_hat: prediction from process model at time k (x_k|k-1).
        z: measurement at time k (z_k).
        P_hat: process model covariance (P_k|k-1, A*P*A_T + Q in linear case).
        R: measurement covariance (R_k).
    Returns:
        mean of fused distribution.
        covariance of fused distribution,
    """
    n = x_hat.shape[-1]
    if x_hat.shape != z.shape:
        raise ValueError(f"x and z are not matching, got {x_hat.shape} and {z.shape}")
    if P_hat.shape != (*x_hat.shape[:-1], n, n):
        raise ValueError(f"Process noise not matching x, expected {(*x_hat.shape[:-1], n, n)}, got {P_hat.shape}")
    if R.shape != (*z.shape[:-1], n, n):
        raise ValueError(f"Measurement noise not matching z, expected {(*x_hat.shape[:-1], n, n)}, got {R.shape}")

    K = P_hat @ torch.inverse(P_hat + R)
    x_f = x_hat + torch.einsum('...ij, ...j->...i', K, z - x_hat)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    P_f = (I - K) @ P_hat @ (I - K).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

    assert torch.all(torch.real(torch.linalg.eig(P_f).eigenvalues) > 0)  # ensure positive definiteness
    P_f = 0.5 * (P_f + P_f.transpose(-1, -2))  # ensure symmetry
    assert torch.allclose(P_f, P_f.transpose(-1, -2))
    return x_f, P_f


def kalman_update_with_distributions(x: Distribution, z: Distribution) -> MultivariateNormal:
    def get_covariance_matrix(d: Distribution):
        if isinstance(d, Normal):
            out = torram.geometry.diag_last(d.variance)
        elif isinstance(d, MultivariateNormal):
            out = d.covariance_matrix
        else:
            raise NotImplementedError(f"Covariance retrieval not implemented for distribution type {type(d)}")
        return out

    P_hat = get_covariance_matrix(x)
    R = get_covariance_matrix(z)
    assert torch.all(torch.real(torch.linalg.eig(P_hat).eigenvalues) > 0)  # ensure positive definiteness
    assert torch.all(torch.real(torch.linalg.eig(R).eigenvalues) > 0)
    x_f, P_f = kalman_update(x.mean, z.mean, P_hat=P_hat, R=R)
    return MultivariateNormal(loc=x_f, covariance_matrix=P_f)
