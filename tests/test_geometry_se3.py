from typing import Tuple

import kornia
import pytest
import torch

from torram.geometry.se3 import (
    geodesic_distance,
    invert_homogeneous_transforms,
    rotation_matrix_from_two_vectors,
)


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


# ------------------------------------------------------------
# Identity case: v1 = v2 → rotation should be identity
# ------------------------------------------------------------
def test_rotation_matrix_from_two_vectors_identity():
    v = torch.tensor([[1.0, 0.0, 0.0]])
    R = rotation_matrix_from_two_vectors(v, v)
    I = torch.eye(3).unsqueeze(0)
    assert torch.allclose(R, I, atol=1e-5, rtol=0)


# ------------------------------------------------------------
# X → Y, Y → Z axis alignment
# ------------------------------------------------------------
def test_rotation_matrix_from_two_vectors_x_to_y():
    v1 = torch.tensor([[1.0, 0.0, 0.0]])  # x-axis
    v2 = torch.tensor([[0.0, 1.0, 0.0]])  # y-axis

    R = rotation_matrix_from_two_vectors(v1, v2)
    v1_rot = (R @ v1.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(v1_rot, v2, atol=1e-5, rtol=0)


def test_rotation_matrix_from_two_vectors_y_to_z():
    v1 = torch.tensor([[0.0, 1.0, 0.0]])  # y-axis
    v2 = torch.tensor([[0.0, 0.0, 1.0]])  # z-axis

    R = rotation_matrix_from_two_vectors(v1, v2)
    v1_rot = (R @ v1.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(v1_rot, v2, atol=1e-5, rtol=0)


# ------------------------------------------------------------
# Random vectors: R * v1 ≈ v2 (direction only)
# ------------------------------------------------------------
def test_rotation_matrix_from_two_vectors_random_vectors():
    torch.manual_seed(0)
    B = 20
    v1 = torch.randn(B, 3) + 1e-6
    v2 = torch.randn(B, 3) + 1e-6

    R = rotation_matrix_from_two_vectors(v1, v2)
    v1_rot = (R @ v1.unsqueeze(-1)).squeeze(-1)

    v1_rot_n = v1_rot / v1_rot.norm(dim=-1, keepdim=True)
    v2_n = v2 / v2.norm(dim=-1, keepdim=True)

    assert torch.allclose(v1_rot_n, v2_n, atol=1e-4, rtol=0)


# ------------------------------------------------------------
# Opposite vectors: v2 = -v1 → 180° rotation
# ------------------------------------------------------------
def test_rotation_matrix_from_two_vectors_opposite_vectors():
    v1 = torch.tensor([[1.0, 0.0, 0.0]])
    v2 = torch.tensor([[-1.0, 0.0, 0.0]])

    R = rotation_matrix_from_two_vectors(v1, v2)
    v1_rot = (R @ v1.unsqueeze(-1)).squeeze(-1)

    # normalize
    v1n = v1_rot / v1_rot.norm()
    v2n = v2 / v2.norm()

    # check they point in opposite directions (dot ~ -1)
    dot = (v1n * v2n).sum()
    assert torch.allclose(dot, torch.tensor(-1.0), atol=1e-4)


# ------------------------------------------------------------
# Batched input shape
# ------------------------------------------------------------
def test_rotation_matrix_from_two_vectors_batch_shape():
    B = 5
    v1 = torch.randn(B, 3)
    v2 = torch.randn(B, 3)
    R = rotation_matrix_from_two_vectors(v1, v2)

    assert R.shape == (B, 3, 3)
