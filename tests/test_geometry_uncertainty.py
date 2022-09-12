import torch
import torram


def test_marginalize_distribution_stddev():
    cov = torch.rand((5, 8, 2, 4, 4))
    stddev_new = torram.geometry.marginalize_distribution(cov, min_variance=1e-8, return_variance=True)
    for i in range(4):
        assert torch.allclose(stddev_new[..., i], cov[..., i, i])


def test_marginalize_distribution_diagonal():
    cov = torch.rand((5, 8, 2, 4, 4))
    cov_new = torram.geometry.marginalize_distribution(cov, min_variance=1e-8)
    for i in range(4):
        assert torch.allclose(cov_new[..., i, i], cov[..., i, i])
    # Check that only the diagonal values are nonzero, i.e. all off-diagonal elements are zero.
    for nz_index in torch.nonzero(cov_new):
        assert nz_index[-2] == nz_index[-1]
