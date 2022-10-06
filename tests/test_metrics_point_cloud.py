import pytest
import torch
import torram


def test_acceleration_zero():
    x = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1, 1).repeat(10, 1, 25, 3)
    x_hat = torch.ones((10, 25, 3), dtype=torch.float32) * 2

    error = torram.metrics.acceleration(x_hat, x)
    assert torch.allclose(error, torch.zeros(10))


def test_acceleration_constant():
    x = torch.tensor([1, 2, 3], dtype=torch.float32).view(1, 3, 1, 1).repeat(10, 1, 25, 3)
    x_hat = torch.ones((10, 25, 3), dtype=torch.float32) * -1

    # accel = x[:, 0] - 2 * x[:, 1] + x[:, 2]
    # accel_hat = x[:, 0] - 2 * x_hat + x[:, 2]
    error_gt = torch.norm(torch.tensor([-2 * (-1.0) + 2 * 2.0] * 3))
    error = torram.metrics.acceleration(x_hat, x)
    assert torch.allclose(error, torch.ones(10) * error_gt)  # same over all N, so average = value


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 3, 3)))
def test_pve_uniform(shape):
    x = torch.ones(shape, dtype=torch.float32)
    y = torch.zeros(shape, dtype=torch.float32)
    pve = torram.metrics.pve(x, y)
    assert torch.allclose(pve, torch.norm(torch.ones(3)))  # euclidean distance (1,1,1) & (0,0,0)


def test_pve_one_different():
    x = torch.rand((1, 6, 3), dtype=torch.float32)
    y = x.clone()
    x[0, 0, 0] += 1
    pve = torram.metrics.pve(x, y)
    assert torch.allclose(pve, torch.ones_like(pve) / x.shape[1])


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 3, 3)))
def test_pve_symmetric(shape):
    x = torch.rand(shape, dtype=torch.float32)
    y = torch.rand(shape, dtype=torch.float32)
    pve_xy = torram.metrics.pve(x, y)
    pve_yx = torram.metrics.pve(y, x)
    assert torch.allclose(pve_xy, pve_yx)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
@pytest.mark.parametrize("scale", (1.0, 2.0, 3.1))
def test_pa_pve_scaling(shape, scale):
    x = torch.rand(shape, dtype=torch.float32)
    y = x * scale
    pa_pve = torram.metrics.pa_pve(x, y)
    assert torch.allclose(pa_pve, torch.zeros_like(pa_pve), atol=1e-5)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
def test_pa_pve_translated(shape):
    batch_size, n, d = shape
    x = torch.rand(shape, dtype=torch.float32)
    y = x - torch.rand((batch_size, 1, d), dtype=torch.float32).repeat(1, n, 1)
    pa_pve = torram.metrics.pa_pve(x, y)
    assert torch.allclose(pa_pve, torch.zeros_like(pa_pve), atol=1e-3)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
def test_pa_pve_rotation(shape):
    batch_size, n, _ = shape
    q3d = torch.rand((batch_size, 3), dtype=torch.float32)
    R = torram.geometry.angle_axis_to_rotation_matrix(q3d)

    x = torch.rand(shape, dtype=torch.float32)
    y = torch.einsum('bij,bkj->bki', R, x)
    pa_pve = torram.metrics.pa_pve(x, y)
    assert torch.allclose(pa_pve, torch.zeros_like(pa_pve), atol=1e-3)


@pytest.mark.parametrize("shape", ((1, 1, 3), (5, 1, 3), (1, 3, 3), (5, 7, 3)))
@pytest.mark.parametrize("scale", (1.0, 2.0, 3.1))
def test_pa_pve_affine(shape, scale):
    batch_size, n, _ = shape
    q3d = torch.rand((batch_size, 3), dtype=torch.float32)
    t = torch.rand((batch_size, 3), dtype=torch.float32)
    T = torram.geometry.pose_to_transformation_matrix(t, q3d)

    x = torch.rand(shape, dtype=torch.float32)
    y = torram.geometry.transform_points(T, x) * scale
    pa_pve = torram.metrics.pa_pve(x, y)
    assert torch.allclose(pa_pve, torch.zeros_like(pa_pve), atol=1e-3)
