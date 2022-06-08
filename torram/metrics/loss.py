import torch


def full_nll_loss(mean: torch.Tensor, target: torch.Tensor, covariance: torch.Tensor,
                  marginalize_cov_norm: bool = False, reduction: str = 'mean'):
    """Negative log-likelihood loss for non-diagonal multivariate Gaussian distributions.

    Generalized form of Gaussian NLL loss as the negative logarithm of a multivariate Gaussian distribution.
    When the covariance is diagonal (or being marginalized) the normalization can be simplified as the
    trace of the logarithm of the diagonal elements, which is both numerically more stable and more efficient.
    https://math.stackexchange.com/questions/2001041/logarithm-of-the-determinant-of-a-positive-definite-matrix

    Args:
        mean: multivariate mean tensor (..., N).
        target: ground-truth values (..., N).
        covariance: full covariance matrix (..., N, N).
        marginalize_cov_norm: marginalize covariance normalization term for efficiency.
        reduction: loss reduction method (mean, sum).
    """
    if mean.shape != target.shape:
        raise ValueError(f"Non-Matching mean and target tensors, got {mean.shape} and {target.shape}")
    if covariance.shape[:-1] != mean.shape or covariance.shape[-2] != covariance.shape[-1]:
        raise ValueError(f"Non-Matching mean and covariance tensors, got {mean.shape} and {covariance.shape}")

    cov_inv = torch.inverse(covariance)
    error = mean - target
    error_term = torch.einsum('...i, ...ij, ...j -> ...', error, cov_inv, error)
    if marginalize_cov_norm:
        norm_term = torch.log(torch.diagonal(covariance, dim1=-2, dim2=-1)).sum(dim=-1)
    else:
        norm_term = torch.log(torch.linalg.det(covariance))

    nll_loss = 0.5 * (error_term + norm_term)  # + k/2*log(2pi)
    if reduction == 'sum':
        result = nll_loss.sum()
    elif reduction == 'mean':
        result = nll_loss.mean()
    else:
        raise ValueError(f"Unknown reduction method {reduction}, choose from [mean, sum]")
    return result
