from typing import Tuple

import kornia
import pytest
import torch

from torram.geometry.se3 import geodesic_distance


def test_geodesic_simple_example():
    x = kornia.geometry.axis_angle_to_rotation_matrix(torch.tensor([[torch.pi / 2, 0, 0]]))
    y = kornia.geometry.axis_angle_to_rotation_matrix(torch.tensor([[torch.pi, 0, 0]]))
    geodesic = geodesic_distance(x, y)
    assert torch.isclose(geodesic, torch.tensor([torch.pi / 2]))


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_none(shape):
    x = kornia.geometry.axis_angle_to_rotation_matrix(torch.rand(shape))
    y = x.clone()
    geodesic = geodesic_distance(x, y)
    assert torch.allclose(geodesic, torch.zeros_like(geodesic), atol=1e-2)


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_symmetric(shape):
    x = kornia.geometry.axis_angle_to_rotation_matrix(torch.rand(shape))
    y = kornia.geometry.axis_angle_to_rotation_matrix(torch.rand(shape))
    geodesic_xy = geodesic_distance(x, y)
    geodesic_yx = geodesic_distance(y, x)
    assert torch.allclose(geodesic_xy, geodesic_yx)


@pytest.mark.parametrize("shape", [(4, 3), (1, 3)])
def test_geodesic_equal(shape: Tuple[int, ...]):
    x = kornia.geometry.axis_angle_to_rotation_matrix(torch.rand(shape, dtype=torch.float64))
    geodesic = geodesic_distance(x, x)
    assert torch.allclose(geodesic, torch.zeros_like(geodesic), atol=0.005)
