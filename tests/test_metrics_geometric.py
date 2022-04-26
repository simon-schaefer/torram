import pytest
import torch
import torram
from typing import Tuple


def geodesic_loss_unbatched(x_hat: torch.Tensor, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    R_diffs = x_hat @ x.permute(0, 2, 1)  # x -> inv(x) = x.T
    traces = R_diffs.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + eps, 1 - eps))


@pytest.mark.parametrize("shape", [(4, 3), (1, 6)])
def test_euclidean_distance_against_zero(shape):
    x = torch.zeros(shape)
    y = torch.rand(shape)
    distance = torram.metrics.euclidean_distance(x, y)
    assert torch.allclose(distance, torch.norm(y, dim=-1))


@pytest.mark.parametrize("shape", [(4, 3), (8, 9)])
def test_euclidean_distance_symmetric(shape):
    x = torch.rand(shape)
    y = torch.rand(shape)
    d_xy = torram.metrics.euclidean_distance(x, y)
    d_yx = torram.metrics.euclidean_distance(y, x)
    assert torch.allclose(d_xy, d_yx)


def test_geodesic_loss_simple_example():
    x = torram.geometry.angle_axis_to_rotation_matrix(torch.tensor([[torch.pi / 2, 0, 0]]))
    y = torram.geometry.angle_axis_to_rotation_matrix(torch.tensor([[torch.pi, 0, 0]]))
    geodesic = torram.metrics.geodesic_loss(x, y)
    assert torch.isclose(geodesic, torch.tensor([torch.pi / 2]))


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_loss_none(shape):
    x = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(shape))
    y = x.clone()
    geodesic = torram.metrics.geodesic_loss(x, y)
    assert torch.allclose(geodesic, torch.zeros_like(geodesic), atol=1e-2)


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_loss_symmetric(shape):
    x = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(shape))
    y = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(shape))
    geodesic_xy = torram.metrics.geodesic_loss(x, y)
    geodesic_yx = torram.metrics.geodesic_loss(y, x)
    assert torch.allclose(geodesic_xy, geodesic_yx)


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_loss_equal(shape: Tuple[int, ...]):
    x = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(shape, dtype=torch.float64))
    geodesic = torram.metrics.geodesic_loss(x, x)
    assert torch.allclose(geodesic, torch.zeros_like(geodesic), atol=0.005)


def test_geodesic_loss_batching():
    x = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(4, 8, 3))
    y = torram.geometry.angle_axis_to_rotation_matrix(torch.rand(4, 8, 3))
    expected = torch.zeros((4, 8))
    for i in range(4):
        expected[i] = geodesic_loss_unbatched(x[i], y[i])
    geodesic = torram.metrics.geodesic_loss(x, y)
    assert torch.allclose(geodesic, expected)
