import math

import kornia
import pytest
import torch

from torram.geometry.points import point2point_distance, point_acceleration


def test_acceleration_zero():
    x = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1, 1).repeat(10, 1, 25, 3)
    acc = point_acceleration(x)
    assert torch.allclose(acc, torch.zeros_like(acc))


def test_acceleration_constant():
    x = torch.ones((10, 50, 3), dtype=torch.float32)
    for k in range(10):
        x[k] = float(k)
    acc = point_acceleration(x)
    assert torch.allclose(acc, torch.zeros_like(acc))


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 3, 3)))
def test_p2p_distance_uniform(shape):
    x = torch.ones(shape, dtype=torch.float32)
    y = torch.zeros(shape, dtype=torch.float32)
    dist = point2point_distance(x, y)
    # euclidean distance (1,1,1) & (0,0,0)
    assert torch.allclose(dist, torch.ones_like(dist) * math.sqrt(3))


def test_p2p_distance_one_different():
    x = torch.rand((1, 6, 3), dtype=torch.float32)
    y = x.clone()
    x[0, 0, 0] += 1
    dist = point2point_distance(x, y)
    assert torch.allclose(dist[1:, 1:], torch.zeros_like(dist[1:, 1:]))
    assert abs(dist[0, 0].item() - 1.0) < 1e-3


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 3, 3)))
def test_p2p_distance_symmetric(shape):
    x = torch.rand(shape, dtype=torch.float32)
    y = torch.rand(shape, dtype=torch.float32)
    pve_xy = point2point_distance(x, y)
    pve_yx = point2point_distance(y, x)
    assert torch.allclose(pve_xy, pve_yx)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
@pytest.mark.parametrize("scale", (1.0, 2.0, 3.1))
def test_p2p_distance_aligned_scaling(shape, scale):
    x = torch.rand(shape, dtype=torch.float32)
    y = x * scale
    pa_dist = point2point_distance(x, y, align=True, align_scale=True)
    assert torch.allclose(pa_dist, torch.zeros_like(pa_dist), atol=1e-5)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
def test_p2p_distance_algined_translated(shape):
    batch_size, n, d = shape
    x = torch.rand(shape, dtype=torch.float32)
    y = x - torch.rand((batch_size, 1, d), dtype=torch.float32).repeat(1, n, 1)
    pa_dist = point2point_distance(x, y, align=True, align_scale=False)
    assert torch.allclose(pa_dist, torch.zeros_like(pa_dist), atol=1e-3)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
def test_p2p_distance_aligned_rotation(shape):
    batch_size, _, _ = shape
    q3d = torch.rand((batch_size, 3), dtype=torch.float32)
    R = kornia.geometry.axis_angle_to_rotation_matrix(q3d)

    x = torch.rand(shape, dtype=torch.float32)
    y = torch.einsum("bij,bkj->bki", R, x)
    pa_dist = point2point_distance(x, y, align=True, align_scale=False)
    assert torch.allclose(pa_dist, torch.zeros_like(pa_dist), atol=1e-3)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
@pytest.mark.parametrize("scale", (1.0, 2.0, 3.1))
def test_p2p_distance_aligned_affine(shape, scale):
    batch_size, n, _ = shape
    q3d = torch.rand((batch_size, 3), dtype=torch.float32)
    t = torch.rand((batch_size, 3, 1), dtype=torch.float32)
    R = kornia.geometry.axis_angle_to_rotation_matrix(q3d)
    T = kornia.geometry.Rt_to_matrix4x4(R, t)

    x = torch.rand(shape, dtype=torch.float32)
    y = kornia.geometry.transform_points(T, x) * scale
    pa_dist = point2point_distance(x, y, align=True, align_scale=True)
    assert torch.allclose(pa_dist, torch.zeros_like(pa_dist), atol=1e-5)
