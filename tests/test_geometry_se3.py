from typing import Tuple

import kornia
import pytest
import torch

from torram.geometry.se3 import geodesic_distance, invert_homogeneous_transforms


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


def make_transform(R=None, t=None):
    """Helper to create a 4x4 homogeneous transform."""
    T = torch.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def test_invert_homogeneous_transform_identity():
    T = torch.eye(4)
    T_inv = invert_homogeneous_transforms(T)
    assert torch.allclose(T_inv, torch.eye(4), atol=1e-6)


def test_invert_homogeneous_transform_random():
    Rs, ts = [], []
    for _ in range(5):
        R, _ = torch.linalg.qr(torch.randn(3, 3))
        Rs.append(R)
        ts.append(torch.randn(3))
    T = torch.stack([make_transform(R, t) for R, t in zip(Rs, ts)], dim=0)

    T_inv = invert_homogeneous_transforms(T)
    I = torch.eye(4).expand_as(T)
    assert torch.allclose(T @ T_inv, I, atol=1e-6)


def test_invert_homogeneous_transform_double_inversion():
    R, _ = torch.linalg.qr(torch.randn(3, 3))
    t = torch.randn(3)
    T = make_transform(R, t)
    T_inv = invert_homogeneous_transforms(T)
    T_double_inv = invert_homogeneous_transforms(T_inv)
    assert torch.allclose(T, T_double_inv, atol=1e-6)


def test_invert_homogeneous_transform_dtype_and_device():
    T = torch.eye(4, dtype=torch.float64, device="cpu")
    T_inv = invert_homogeneous_transforms(T)
    assert T_inv.dtype == torch.float64
    assert T_inv.device == T.device
    assert torch.allclose(T_inv, T)
