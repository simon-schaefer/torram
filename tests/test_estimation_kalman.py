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

    x_f, P_f = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, (x_hat + z) / 2)
    expected = torch.mul(0.5, torram.geometry.eye(shape))
    assert torch.allclose(P_f, expected, atol=1e-3)


@pytest.mark.parametrize("shape", ((1, ), (8, 2), (8, 1, 3)))
def test_z_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = torram.geometry.eye(shape) * 1000
    R = torram.geometry.eye(shape) * 0.1

    x_f, P_f = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, z, atol=1e-3)
    assert torch.allclose(P_f, R, atol=1e-3)


@pytest.mark.parametrize("shape", ((1, ), (8, 2), (8, 1, 3)))
def test_x_certain(shape: Tuple[int, ...]):
    x_hat = torch.rand(shape)
    z = torch.rand(shape)
    P_hat = torram.geometry.eye(shape) * 0.1
    R = torram.geometry.eye(shape) * 1000

    x_f, P_f = torram.estimation.kalman_update(x_hat, z, P_hat=P_hat, R=R)
    assert torch.allclose(x_f, x_hat, atol=1e-3)
    assert torch.allclose(P_f, P_hat, atol=1e-3)
