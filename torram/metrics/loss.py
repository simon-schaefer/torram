import itertools
import numpy as np
import torch


__all__ = ['full_nll_loss',
           'contrastive_loss']


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

    cov_inv = torch.inverse(covariance)  # TODO: use torch.linalg.solve(covariance, error) instead .inverse(...)
    error = mean - target
    error_term = torch.einsum('...i, ...ij, ...j -> ...', error, cov_inv, error)
    if marginalize_cov_norm:
        cov_norm = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    else:
        cov_norm = torch.linalg.det(covariance)
    norm_term = torch.log(torch.clamp(cov_norm, min=1e-6))

    nll_loss = 0.5 * (error_term + norm_term)  # + k/2*log(2pi)
    if reduction == 'sum':
        result = nll_loss.sum()
    elif reduction == 'mean':
        result = nll_loss.mean()
    else:
        raise ValueError(f"Unknown reduction method {reduction}, choose from [mean, sum]")

    return result


def contrastive_loss(
    y_hat: torch.Tensor,
    anchors: torch.Tensor,
    anchor_margin: float = 0.0,
    margin: float = 0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Compute the contrastive loss (also ranking loss) of the prediction `y_hat` given a target for the
    prediction's ordering. The contrastive loss targets the following ordering:

        y_hat[i] > y_hat[j]     if anchors[i] > anchors[j]
        y_hat[j] >= y_hat[i]    otherwise

    Args:
        y_hat: model predictions (N,).
        anchors: reference values for prediction ordering (N,).
        anchor_margin: minimal difference of the anchors to be included.
        margin: minimal difference of the predictions to be included.
        reduction: specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """
    if y_hat.shape != anchors.shape:
        raise ValueError(f"Non-Matching shape of y_hat and anchors, got {y_hat.shape} and {anchors.shape}")
    if len(y_hat.shape) != 1:
        raise ValueError(f"Expected flat input tensor, got {y_hat.shape}")
    pairs_ij = np.array([ij for ij in itertools.combinations(list(range(len(y_hat))), 2)])
    labels_ij = [1 if anchors[i] > anchors[j] else -1 for i, j in pairs_ij]
    labels_ij = torch.tensor(labels_ij, device=anchors.device, dtype=anchors.dtype)
    is_included = [k for k, (i, j) in enumerate(pairs_ij) if abs(anchors[i] - anchors[j]) > anchor_margin]
    return torch.nn.functional.margin_ranking_loss(
        input1=y_hat[pairs_ij[is_included, 0]],
        input2=y_hat[pairs_ij[is_included, 1]],
        target=labels_ij[is_included],
        margin=margin,
        reduction=reduction
    )
