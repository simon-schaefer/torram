import kornia
import pytest
import torch
import torram


@pytest.mark.parametrize("shape", [(4, 2), (4, 1, 3, 1), (8,)])
def test_inverse_transformation(shape):
    q = torch.rand((*shape, 3), dtype=torch.float32)
    t = torch.rand((*shape, 3), dtype=torch.float32)

    T = torch.zeros((*shape, 4, 4), dtype=torch.float32)
    T[..., :3, :3] = kornia.geometry.angle_axis_to_rotation_matrix(q.view(-1, 3)).view((*shape, 3, 3))
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    T_inv = torram.geometry.inverse_transformation(T)
    T_hat = torram.geometry.inverse_transformation(T_inv)
    assert torch.allclose(T_hat, T, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3), (5, 6, 3)])
def test_inverse_quaternion(shape):
    x3d = torch.rand(shape)
    q = torram.geometry.angle_axis_to_quaternion(x3d)
    q_inv = torram.geometry.inverse_quaternion(q)
    q_unit = torram.geometry.angle_axis_to_quaternion(torch.zeros(shape))
    assert torch.allclose(torram.geometry.multiply_quaternion(q, q_inv), q_unit, atol=1e-6)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_angle_axis_to_rotation_matrix(shape):
    x3d = torch.rand(shape)
    R_hat = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    x3d_flat = torch.flatten(x3d, end_dim=-2)
    R = kornia.geometry.angle_axis_to_rotation_matrix(x3d_flat)
    assert torch.allclose(torch.flatten(R_hat, end_dim=-3), R)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_and_angle_axis(shape):
    x3d = torch.rand(shape)
    x6d = torram.geometry.angle_axis_to_rotation_6d(x3d)
    x3d_hat = torram.geometry.rotation_6d_to_axis_angle(x6d)
    assert torch.allclose(x3d_hat, x3d_hat)


def test_rotation_matrix_to_quaternion_grad_not_nan():
    R = torch.tensor([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]], dtype=torch.float32, requires_grad=True)
    R2 = R * 0.99999
    q = kornia.geometry.rotation_matrix_to_quaternion(R2)
    grad, = torch.autograd.grad(q.sum(), R)
    assert not torch.any(torch.isnan(grad))


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_rotation_matrix(shape):
    x3d = torch.rand(shape)
    R = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    x6d_hat = torram.geometry.rotation_matrix_to_rotation_6d(R)
    R_hat = torram.geometry.rotation_6d_to_rotation_matrix(x6d_hat)
    assert torch.allclose(R_hat, R, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_rotation_6d_cross_transforms(shape):
    x3d = torch.rand(shape)
    x6d = torram.geometry.angle_axis_to_rotation_6d(x3d)
    R = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    x6d_hat = torram.geometry.rotation_matrix_to_rotation_6d(R)
    assert torch.allclose(x6d_hat, x6d)


def test_rotation_matrix_to_quaternion_1d():
    x3d = torch.rand(3)
    R_hat = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    assert R_hat.shape == (3, 3)


def test_rotation_matrix_to_quaternion_2d():
    x3d = torch.rand(3)
    R = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    q_hat = torram.geometry.rotation_matrix_to_quaternion(R)
    assert q_hat.shape == (4, )

