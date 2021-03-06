import kornia
import pytest
import torch
import torram

from torram.geometry import pose_to_transformation_matrix
from torram.geometry import jacobians as torram_jacobians
from typing import Tuple


def get_random_transform(shape, dtype=torch.float64):
    t = torch.rand((*shape, 3), dtype=dtype)
    q = torch.rand((*shape, 4), dtype=dtype)
    return pose_to_transformation_matrix(t, q)


def T_from_q(q: torch.Tensor) -> torch.Tensor:
    t = torch.zeros((*q.shape[:-1], 3), dtype=q.dtype, device=q.device)
    return torram.geometry.pose_to_transformation_matrix(t, q)


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()


def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data
    return torch.transpose(jacobian, dim0=0, dim1=1)


@pytest.mark.parametrize("shape", ((3, ), (8, 3), (1, 3), (5, 8, 3)))
def test_T_wrt_t(shape, delta: float = 1e-6):
    def T_from_t(t: torch.Tensor) -> torch.Tensor:
        return torram.geometry.pose_to_transformation_matrix(t, torch.zeros_like(t))

    x = torch.rand(shape, dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_t(x)
    T = T_from_t(x)
    for i in range(3):
        x_ = x.clone()
        x_[..., i] += delta
        T_ = T_from_t(x_)
        J = (T_ - T) / delta
        assert torch.allclose(J_hat[..., i], J, atol=1e-5)


@pytest.mark.parametrize("shape", ((3, ), (8, 3), (1, 3), (5, 8, 3)))
def test_T_wrt_q3d(shape: Tuple[int, ...], delta: float = 1e-6):
    x = torch.rand(shape, dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_q3d(x)
    T = T_from_q(x)
    for i in range(3):
        x_ = x.clone()
        x_[..., i] += delta
        T_ = T_from_q(x_)
        J = (T_ - T) / delta
        assert torch.allclose(J_hat[..., i], J, atol=1e-5)


def test_T_wrt_q3d_zeros():
    x = torch.zeros((3, ), dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_q3d(x)
    assert not torch.any(torch.isnan(J_hat))


@pytest.mark.parametrize("shape", ((4, ), (8, 4), (1, 4), (5, 8, 4)))
def test_T_wrt_q4d(shape: Tuple[int, ...], delta: float = 1e-6):
    x = torch.rand(shape, dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_q4d(x)
    T = T_from_q(x)
    for i in range(4):
        x_ = x.clone()
        x_[..., i] += delta
        T_ = T_from_q(x_)
        J = (T_ - T) / delta
        assert torch.allclose(J_hat[..., i], J, atol=1e-5)


@pytest.mark.parametrize("shape", ((4, 4), (8, 4, 4), (1, 4, 4), (5, 8, 4, 4)))
def test_t_wrt_T(shape, delta: float = 1e-6):
    x = get_random_transform(shape[:-2])
    J_hat = torram_jacobians.t_wrt_T(x)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[..., i, j] += delta
            J = (x_[..., :3, 3] - x[..., :3, 3]) / delta
            assert torch.allclose(J_hat[..., i, j], J, atol=1e-5)


@pytest.mark.parametrize("shape", ((4, 4), (8, 4, 4), (1, 4, 4), (5, 8, 4, 4)))
def test_q4d_wrt_T(shape, delta: float = 1e-6):
    x = get_random_transform(shape[:-2])
    q = torram.geometry.rotation_matrix_to_quaternion(x[..., :3, :3].contiguous(), eps=1e-12)
    J_hat = torram_jacobians.q4d_wrt_T(x)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[..., i, j] += delta
            q_ = kornia.geometry.rotation_matrix_to_quaternion(x_[..., :3, :3].contiguous(), eps=1e-12)
            J = (q_ - q) / delta
            assert torch.allclose(J_hat[..., i, j], J, atol=1e-4)


def test_q3d_wrt_T():
    """Numerical differentiation is hard here as the rotation matrix R is not valid after adding
    an increment (norm != 1). Therefore, using autograd here."""
    x = get_random_transform((1, ), dtype=torch.float32)
    x.requires_grad = True
    q = torram.geometry.rotation_matrix_to_angle_axis(x[..., :3, :3].contiguous(), epsilon=0)
    J_hat = torram_jacobians.q3d_wrt_T(x, epsilon=0)
    J_autograd = compute_jacobian(x, q)
    assert torch.allclose(J_hat, J_autograd, atol=1e-4)


@pytest.mark.parametrize("shape", ((4, 4), (8, 4, 4), (1, 4, 4), (5, 8, 4, 4)))
def test_T_inv_wrt_T(shape, delta: float = 1e-6):
    x = get_random_transform(shape[:-2])
    J_hat = torram_jacobians.T_inv_wrt_T(x).view(*shape[:-2], 16, 4, 4)
    x_inv = torram.geometry.inverse_transformation(x)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[..., i, j] += delta
            x_inv_ = torram.geometry.inverse_transformation(x_)
            J = (x_inv_ - x_inv) / delta
            J = torch.flatten(J, start_dim=-2)
            assert torch.allclose(J_hat[..., i, j], J)
