import torch
from jaxtyping import Float
from torch.distributions import MultivariateNormal

from torram.utils.ops import diag_last


def marginalize_distribution(
    cov: Float[torch.Tensor, "... N N"],
    min_variance: float = 0.001,
    return_variance: bool = False,
) -> Float[torch.Tensor, "... N N"]:
    assert min_variance > 0

    var_marginal = torch.diagonal(cov, dim1=-2, dim2=-1)
    var_marginal = torch.clamp_min(var_marginal, min_variance)
    if return_variance:
        return var_marginal
    return diag_last(var_marginal)


def cov_error_propagation(
    covariance_matrix: Float[torch.Tensor, "B N N"],
    Jx: Float[torch.Tensor, "B N N"],
    *,
    square_form: bool = False,
    nan_to_zero: bool = False,
) -> Float[torch.Tensor, "B N N"]:
    """Covariance error propagation.

    For a covariance matrix C which is transformed by some transform T(x) with jacobian J(x) = dT/dx, the
    transformed covariance matrix C' is:

    C' = J * C * J^T

    For numerical stability the square-root form of this equation can be used:

    C* = J * sqrt(C)
    C' = C* * (C*)^T

    Args:
        covariance_matrix: covariance matrix of source distribution.
        Jx: jacobian dT/dx.
        square_form: use the square-root form or the standard form to calculate C'.
        nan_to_zero: convert nan gradients to zeros.
    """
    if nan_to_zero:
        Jx = torch.nan_to_num(Jx, nan=0)
    if square_form:
        covariance_matrix = torch.clamp_min(covariance_matrix, min=1e-6)  # for numerical stability
        cov_ = torch.matmul(Jx, torch.sqrt(covariance_matrix))
        return torch.matmul(cov_, cov_.transpose(-1, -2))

    return torch.einsum("...ij,...jk,...kl->...il", Jx, covariance_matrix, Jx.transpose(-1, -2))


def cov_error_propagation_mv(
    x: MultivariateNormal,
    Jx: Float[torch.Tensor, "B N N"],
    square_form: bool = False,
    nan_to_zero: bool = False,
) -> MultivariateNormal:
    """Covariance error propagation for MultivariateNormal distribution."""
    x_cov = x.covariance_matrix
    cov_out = cov_error_propagation(x_cov, Jx, square_form=square_form, nan_to_zero=nan_to_zero)
    return MultivariateNormal(loc=x.mean, covariance_matrix=cov_out)


def full_nll_loss(
    mean: torch.Tensor,
    target: torch.Tensor,
    covariance: torch.Tensor,
    marginalize_cov_norm: bool = False,
    reduction: str = "mean",
):
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
    assert mean.shape == target.shape
    assert covariance.shape[-1] == covariance.shape[-2]
    assert covariance.shape[:-1] == mean.shape

    cov_inv = torch.inverse(
        covariance
    )  # TODO: use torch.linalg.solve(covariance, error) instead .inverse(...)
    error = mean - target
    error_term = torch.einsum("...i, ...ij, ...j -> ...", error, cov_inv, error)
    if marginalize_cov_norm:
        cov_norm = torch.diagonal(covariance, dim1=-2, dim2=-1).sum(dim=-1)
    else:
        cov_norm = torch.linalg.det(covariance)
    norm_term = torch.log(torch.clamp(cov_norm, min=1e-6))

    nll_loss = 0.5 * (error_term + norm_term)  # + k/2*log(2pi)
    if reduction == "sum":
        result = nll_loss.sum()
    elif reduction == "mean":
        result = nll_loss.mean()
    else:
        raise ValueError(f"Unknown reduction method {reduction}, choose from [mean, sum]")

    return result
