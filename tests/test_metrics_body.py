import torch
import torram


def test_mpjpe_symmetric():
    x = torch.rand((4, 23, 3), dtype=torch.float32)
    y = torch.rand((4, 23, 3), dtype=torch.float32)

    mpjpe_xy = torram.metrics.mpjpe(x, y)
    mpjpe_yx = torram.metrics.mpjpe(y, x)
    assert torch.allclose(mpjpe_xy, mpjpe_yx)


def test_mpjpe_distance_zero():
    x = torch.rand((4, 23, 3), dtype=torch.float32)
    y = x.clone()
    mpjpe = torram.metrics.mpjpe(x, y)
    assert torch.allclose(mpjpe, torch.zeros_like(mpjpe))


def test_mpjpe_translated_translated():
    x = torch.zeros((4, 23, 3), dtype=torch.float32)
    y = x.clone() + torch.rand((4, 1, 3), dtype=torch.float32).repeat(1, 23, 1)
    mpjpe = torram.metrics.mpjpe(x, y)
    assert torch.allclose(mpjpe, torch.zeros_like(mpjpe))


def test_mpjpe_uniform_distance():
    x = torch.ones((4, 23, 3), dtype=torch.float32)
    y = torch.ones((4, 23, 3), dtype=torch.float32) * 4
    mpjpe = torram.metrics.mpjpe(x, y)
    assert torch.allclose(mpjpe, torch.zeros_like(mpjpe))


def test_pa_mpjpe_affine():
    q3d = torch.rand((4, 3), dtype=torch.float32)
    t = torch.rand((4, 3), dtype=torch.float32)
    T = torram.geometry.pose_to_transformation_matrix(t, q3d)

    x = torch.rand((4, 23, 3), dtype=torch.float32)
    y = torram.geometry.transform_points(T, x) * 3.11
    pa_mpjpe = torram.metrics.pa_mpjpe(x, y)
    assert torch.allclose(pa_mpjpe, torch.zeros_like(pa_mpjpe), atol=1e-6)
