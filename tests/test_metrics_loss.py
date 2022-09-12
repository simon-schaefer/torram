import torch
import torram


def test_full_nll_loss_against_torch():
    mean = torch.rand((4, 3))
    target = torch.rand((4, 3))
    variance = torch.rand((4, 3))
    covariance = torram.geometry.diag_last(variance)

    loss = torch.nn.functional.gaussian_nll_loss(mean, target, var=variance, eps=0, reduction='sum')
    loss_hat = torram.metrics.full_nll_loss(mean, target, covariance, reduction='sum')
    assert torch.allclose(loss_hat, loss)


def test_contrastive_loss():
    y_hat = torch.arange(10, dtype=torch.float32)
    y_hat[4] = 10.0
    target = torch.arange(10)
    loss = torram.metrics.contrastive_loss(y_hat, target, reduction='sum')
    assert float(loss) == (5 + 4 + 3 + 2 + 1)  # all elements above 4 => 10 - 5, 10 - 6, ..., 10 -9
