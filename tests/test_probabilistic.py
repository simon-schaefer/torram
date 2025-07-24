from typing import Tuple

import pytest
import torch

from torram.probabilistic.gmm import GaussianMixtureModel
from torram.probabilistic.kalman import kalman_update
from torram.probabilistic.ops import full_nll_loss, marginalize_distribution
from torram.utils.ops import diag_last, eye


def create_random_gmm(num_modes: int, n_dim: int, means=None, covariances=None, weights=None):
    if means is None:
        means = torch.rand((num_modes, n_dim))
    if covariances is None:
        covariances = diag_last(torch.rand((num_modes, n_dim)))
    if weights is None:
        weights = torch.rand(num_modes)
    return GaussianMixtureModel(means, covariances, weights)


@pytest.mark.parametrize("shape", ((1,), (8, 2), (8, 1, 3)))
def test_kalman_equal_uncertainty(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = eye(shape)
    R = eye(shape)

    x_f, P_f = kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, (x_hat + z) / 2)
    expected = torch.mul(0.5, eye(shape))
    assert torch.allclose(P_f, expected, atol=1e-3)


@pytest.mark.parametrize("shape", ((1,), (8, 2), (8, 1, 3)))
def test_kalman_z_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = eye(shape) * 1000
    R = eye(shape) * 0.1

    x_f, P_f = kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, z, atol=1e-3)
    assert torch.allclose(P_f, R, atol=1e-3)


@pytest.mark.parametrize("shape", ((1,), (8, 2), (8, 1, 3)))
def test_kalman_x_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = eye(shape) * 0.1
    R = eye(shape) * 1000

    x_f, P_f = kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, x_hat, atol=1e-3)
    assert torch.allclose(P_f, P_hat, atol=1e-3)


def test_gmm_log_prob():
    gmm = create_random_gmm(4, 69)
    samples = torch.rand((2, 69))
    log_probs = gmm.log_prob(samples)
    assert log_probs.shape == (2,)
    assert torch.all(log_probs < 0)


def test_gmm_get_mean():
    means = torch.rand((4, 69))
    gmm = create_random_gmm(4, 69, means=means)
    assert torch.allclose(means, gmm.mean)


def test_gmm_get_covariances():
    covariances = diag_last(torch.rand((4, 69)))
    gmm = create_random_gmm(4, 69, covariances=covariances)
    assert torch.allclose(covariances, gmm.variance)


def test_marginalize_distribution_stddev():
    cov = torch.rand((5, 8, 2, 4, 4))
    stddev_new = marginalize_distribution(cov, min_variance=1e-8, return_variance=True)
    for i in range(4):
        assert torch.allclose(stddev_new[..., i], cov[..., i, i])


def test_marginalize_distribution_diagonal():
    cov = torch.rand((5, 8, 2, 4, 4))
    cov_new = marginalize_distribution(cov, min_variance=1e-8)
    for i in range(4):
        assert torch.allclose(cov_new[..., i, i], cov[..., i, i])
    # Check that only the diagonal values are nonzero, i.e. all off-diagonal elements are zero.
    for nz_index in torch.nonzero(cov_new):
        assert nz_index[-2] == nz_index[-1]


def test_full_nll_loss_against_torch():
    mean = torch.rand((4, 3))
    target = torch.rand((4, 3))
    variance = torch.rand((4, 3))
    covariance = diag_last(variance)

    loss = torch.nn.functional.gaussian_nll_loss(mean, target, var=variance, eps=0, reduction="sum")
    loss_hat = full_nll_loss(mean, target, covariance, reduction="sum")
    assert torch.allclose(loss_hat, loss)
