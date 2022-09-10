import kornia
import pytest
import torch
import torram

from torch.nn.functional import normalize
from typing import Tuple


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


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 5, 3)])
def test_multiply_angle_axis(shape: Tuple[int, ...]):
    a = torch.rand(shape)
    b = torch.rand(shape)
    c_hat = torram.geometry.multiply_angle_axis(a, b, eps=0)

    Ra = torram.geometry.angle_axis_to_rotation_matrix(a)
    Rb = torram.geometry.angle_axis_to_rotation_matrix(b)
    c = kornia.geometry.rotation_matrix_to_angle_axis(Ra @ Rb)
    assert torch.allclose(c_hat, c)


def test_multiply_angle_axis_zeros():
    a = torch.zeros((4, 3))
    b = torch.rand((4, 3))
    c_hat = torram.geometry.multiply_angle_axis(a, b)
    assert not torch.any(torch.isnan(c_hat))


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3)])
def test_angle_axis_to_rotation_matrix(shape):
    x3d = torch.rand(shape)
    R_hat = torram.geometry.angle_axis_to_rotation_matrix(x3d)
    x3d_flat = torch.flatten(x3d, end_dim=-2)
    R = kornia.geometry.angle_axis_to_rotation_matrix(x3d_flat)
    assert torch.allclose(torch.flatten(R_hat, end_dim=-3), R)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3), (3, )])
def test_rotation_6d_to_angle_axis(shape):
    x3d = torch.rand(shape)
    x6d = torram.geometry.angle_axis_to_rotation_6d(x3d)
    x3d_hat = torram.geometry.rotation_6d_to_axis_angle(x6d)
    assert torch.allclose(x3d_hat, x3d_hat)


@pytest.mark.parametrize("shape", [(4, 3), (4, 1, 3, 3), (3, )])
def test_rotation_matrix_to_angle_axis_against_kornia(shape):
    q3d = torch.rand(shape)
    rotation_matrix = torram.geometry.angle_axis_to_rotation_matrix(q3d)

    q3d_kornia = kornia.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_kornia, q3d)  # just to check kornia implementation
    q3d_hat = torram.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_hat, q3d)


def test_rotation_matrix_singularity_eye():
    rotation_matrix = torch.stack([torch.eye(3),
                                   torram.geometry.angle_axis_to_rotation_matrix(torch.rand(3))], dim=0)
    q3d_hat = torram.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    q3d_kornia = kornia.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_hat, q3d_kornia)


def test_rotation_matrix_singularity_180_x():
    rotation_matrix = torch.tensor([[1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, -1]], dtype=torch.float64)
    q3d_hat = torram.geometry.rotation_matrix_to_angle_axis(rotation_matrix, epsilon=1e-8)
    q3d_kornia = kornia.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_hat, q3d_kornia)


def test_rotation_matrix_singularity_180_y():
    rotation_matrix = torch.tensor([[-1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, -1]], dtype=torch.float64)
    q3d_hat = torram.geometry.rotation_matrix_to_angle_axis(rotation_matrix, epsilon=1e-8)
    q3d_kornia = kornia.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_hat, q3d_kornia)


def test_rotation_matrix_singularity_180_z():
    rotation_matrix = torch.tensor([[-1, 0, 0],
                                    [0, -1, 0],
                                    [0, 0, 1]], dtype=torch.float64)
    q3d_hat = torram.geometry.rotation_matrix_to_angle_axis(rotation_matrix, epsilon=1e-8)
    q3d_kornia = kornia.geometry.rotation_matrix_to_angle_axis(rotation_matrix)
    assert torch.allclose(q3d_hat, q3d_kornia)


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


@pytest.mark.parametrize("shape", [(4, 4), (4, 1, 3, 4), (4, )])
def test_quaterion_to_rotation_matrix(shape):
    q = torch.rand(shape)
    R = torram.geometry.quaternion_to_rotation_matrix(q)
    q_hat = torram.geometry.rotation_matrix_to_quaternion(R)
    assert torch.allclose(normalize(q_hat, dim=-1), normalize(q, dim=-1))


@pytest.mark.parametrize("shape", [(4, 4), (4, 1, 3, 4), (4, )])
def test_quaternion_to_angle_axis(shape):
    q = torch.rand(shape)
    a = torram.geometry.quaternion_to_angle_axis(q)
    q_hat = torram.geometry.angle_axis_to_quaternion(a)
    assert torch.allclose(normalize(q_hat, dim=-1), normalize(q, dim=-1))


@pytest.mark.parametrize("data2d, values", [
    (torch.ones(1, 2, 2), torch.ones(1,)),
    (torch.ones(5, 2, 2), torch.ones(5,)),
    (torch.arange(4).view(1, 2, 2), torch.ones(1, ) * 1.5)
])
def test_interpolate2d(data2d: torch.Tensor, values: torch.Tensor):
    y = torram.geometry.interpolate2d(data2d, points=torch.tensor([[0.5, 0.5]]))
    assert torch.allclose(y, values)
