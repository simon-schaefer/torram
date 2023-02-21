import torch

from torram.distributions import GaussianMixtureModel
from torram.geometry import diag_last


def create_random_gmm(num_modes: int, n_dim: int, means=None, covariances=None, weights=None):
    if means is None:
        means = torch.rand((num_modes, n_dim))
    if covariances is None:
        covariances = diag_last(torch.rand((num_modes, n_dim)))
    if weights is None:
        weights = torch.rand(num_modes)
    return GaussianMixtureModel(means, covariances, weights)


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
