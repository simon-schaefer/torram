import torch

from torch.distributions import constraints, Distribution, MultivariateNormal

__all__ = ['GaussianMixtureModel']


class GaussianMixtureModel(Distribution):
    arg_constraints = {}
    support = constraints.real

    def __init__(self, means: torch.Tensor, covariances: torch.Tensor, weights: torch.Tensor, validate_args=None):
        assert means.ndim == 2  # (num_modes, num_dim)
        self.num_modes, nd = means.shape
        assert covariances.shape == (self.num_modes, nd, nd)
        assert weights.shape == (self.num_modes, )

        self.num_modes = len(weights)
        self.modes = [MultivariateNormal(means[i], covariances[i]) for i in range(self.num_modes)]
        self.weights = weights

        batch_shape = self.weights.size()
        super(GaussianMixtureModel, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        raise NotImplementedError

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor):
        assert isinstance(value, torch.Tensor)
        log_probs = torch.zeros((*value.shape[:-1], self.num_modes), device=value.device, dtype=value.dtype)
        for i in range(self.num_modes):
            log_probs[..., i] = self.weights[i] * self.modes[i].log_prob(value)
        return torch.max(log_probs, dim=-1).values

    def cdf(self, value):
        raise NotImplementedError

    def icdf(self, value):
        raise NotImplementedError

    def enumerate_support(self, expand=True):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    @property
    def mode(self):
        return self.mean

    @property
    def mean(self):
        return torch.stack([m.mean for m in self.modes], dim=0)

    @property
    def variance(self):
        return torch.stack([m.covariance_matrix for m in self.modes], dim=0)
