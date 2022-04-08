import torch
import torram

from torch.distributions import Distribution, Normal, MultivariateNormal

__all__ = [
    'kalman_update',
    'kalman_update_with_distributions'
]


def kalman_update(x_hat: torch.Tensor, z: torch.Tensor, P_hat: torch.Tensor, R: torch.Tensor) -> MultivariateNormal:
    """Kalman update equations for merging predictions from the process model (x) and measurement (z).

    This implementation assumes that the output of the process model (x) and the measurement (z) are describing
    the same quantity, i.e. that the matrix H with x_z = H * z is the identity matrix.

    Args:
        x_hat: prediction from process model at time k.
        z: measurement at time k.
        P_hat: process model covariance (A*P*A_T + Q in linear case).
        R: measurement covariance
    """
    n = x_hat.shape[-1]
    if x_hat.shape != z.shape:
        raise ValueError(f"x and z are not matching, got {x_hat.shape} and {z.shape}")
    if P_hat.shape != (*x_hat.shape[:-1], n, n):
        raise ValueError(f"Process noise not matching x, expected {(*x_hat.shape[:-1], n, n)}, got {P_hat.shape}")
    if R.shape != (*z.shape[:-1], n, n):
        raise ValueError(f"Measurement noise not matching z, expected {(*x_hat.shape[:-1], n, n)}, got {R.shape}")

    K = torch.matmul(P_hat, torch.inverse(P_hat + R))
    x_f = x_hat + torch.einsum('...ij, ...j->...i', K, z - x_hat)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    P_f = torch.matmul(I - K, P_hat)
    return MultivariateNormal(loc=x_f, covariance_matrix=P_f)


def kalman_update_with_distributions(x: Distribution, z: Distribution) -> MultivariateNormal:
    def get_covariance_matrix(d: Distribution):
        if isinstance(d, Normal):
            return torram.geometry.diag_last(d.variance)
        elif isinstance(d, MultivariateNormal):
            return d.covariance_matrix
        else:
            raise NotImplementedError(f"Covariance retrieval not implemented for distribution type {type(d)}")

    P_hat = get_covariance_matrix(x)
    R = get_covariance_matrix(z)
    return kalman_update(x.mean, z.mean, P_hat=P_hat, R=R)
