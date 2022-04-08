import pytest
import torch
import torram

from typing import Tuple


@pytest.mark.parametrize("shape", ((1, ), (8, 2), (8, 1, 3)))
def test_equal_uncertainty(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = torram.geometry.eye(shape)
    R = torram.geometry.eye(shape)

    x = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x.mean, (x_hat + z) / 2)
    expected = torram.geometry.eye(shape) * 0.5
    assert torch.allclose(x.covariance_matrix, expected)


@pytest.mark.parametrize("shape", ((1, ), (8, 2), (8, 1, 3)))
def test_z_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = torram.geometry.eye(shape) * 1000
    R = torram.geometry.eye(shape) * 0.1

    x = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x.mean, z, atol=1e-3)
    assert torch.allclose(x.covariance_matrix, R, atol=1e-3)


@pytest.mark.parametrize("shape", ((1, ), (8, 2), (8, 1, 3)))
def test_x_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = torram.geometry.eye(shape) * 0.1
    R = torram.geometry.eye(shape) * 1000

    x = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x.mean, x_hat, atol=1e-3)
    assert torch.allclose(x.covariance_matrix, P_hat, atol=1e-3)
