from typing import Tuple

import torch
from jaxtyping import Float
from torch.distributions import MultivariateNormal

__all__ = ["kalman_update", "kalman_update_mvn"]


def kalman_update(
    x_hat: Float[torch.Tensor, "B N"],
    z: Float[torch.Tensor, "B N"],
    P_hat: Float[torch.Tensor, "B N N"],
    R: Float[torch.Tensor, "B N N"],
    check_positive_definite: bool = True,
) -> Tuple[Float[torch.Tensor, "B N"], Float[torch.Tensor, "B N"]]:
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
    if check_positive_definite:
        assert torch.all(torch.real(torch.linalg.eig(P_hat).eigenvalues) > 0)
        assert torch.all(torch.real(torch.linalg.eig(R).eigenvalues) > 0)

    n = x_hat.shape[-1]
    eye = torch.eye(n, device=x_hat.device, dtype=x_hat.dtype)

    K = P_hat @ torch.inverse(P_hat + R)
    x_f = x_hat + torch.einsum("...ij, ...j->...i", K, z - x_hat)
    P_f = (eye - K) @ P_hat @ (eye - K).transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

    P_f = 0.5 * (P_f + P_f.transpose(-1, -2))  # ensure symmetry
    assert torch.allclose(P_f, P_f.transpose(-1, -2))
    return x_f, P_f


def kalman_update_mvn(x: MultivariateNormal, z: MultivariateNormal) -> MultivariateNormal:
    x_f, P_f = kalman_update(x.mean, z.mean, P_hat=x.covariance_matrix, R=z.covariance_matrix)
    return MultivariateNormal(loc=x_f, covariance_matrix=P_f)
