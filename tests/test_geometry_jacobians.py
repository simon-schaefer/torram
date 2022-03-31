import kornia
import torch

import torram.geometry
from torram.geometry import pose_to_transformation_matrix
from torram.geometry import jacobians as torram_jacobians


def get_random_transform():
    t = torch.rand(3, dtype=torch.float64)
    q = torch.rand(4, dtype=torch.float64)
    return pose_to_transformation_matrix(t, q)


def test_T_wrt_t(delta: float = 1e-6):
    def T_from_t(t: torch.Tensor) -> torch.Tensor:
        out = torch.eye(4, dtype=t.dtype, device=t.device)
        out[:3, 3] = t
        return out

    x = torch.rand(3, dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_t(x)
    T = T_from_t(x)
    for i in range(3):
        x_ = x.clone()
        x_[i] += delta
        T_ = T_from_t(x_)
        J = (T_ - T) / delta
        assert torch.allclose(J_hat[:, i].view(4, 4), J, atol=1e-5)


def test_T_wrt_q4d(delta: float = 1e-6):
    def T_from_q(q: torch.Tensor) -> torch.Tensor:
        out = torch.eye(4, dtype=q.dtype, device=q.device)
        out[:3, :3] = torram.geometry.quaternion_to_rotation_matrix(q)
        return out

    x = torch.rand(4, dtype=torch.float64)
    J_hat = torram_jacobians.T_wrt_q4d(x)
    T = T_from_q(x)
    for i in range(4):
        x_ = x.clone()
        x_[i] += delta
        T_ = T_from_q(x_)
        J = (T_ - T) / delta
        assert torch.allclose(J_hat[:, i].view(4, 4), J, atol=1e-5)


def test_t_wrt_T(delta: float = 1e-6):
    x = get_random_transform()
    J_hat = torram_jacobians.t_wrt_T(x).view(3, 4, 4)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[i, j] += delta
            J = (x_[:3, 3] - x[:3, 3]) / delta
            assert torch.allclose(J_hat[:, i, j], J, atol=1e-5)


def test_q4d_wrt_T(delta: float = 1e-6):
    x = get_random_transform()
    q = kornia.geometry.rotation_matrix_to_quaternion(x[:3, :3].contiguous())
    J_hat = torram_jacobians.q4d_wrt_T(x).view(4, 4, 4)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[i, j] += delta
            q_ = kornia.geometry.rotation_matrix_to_quaternion(x_[:3, :3].contiguous())
            J = (q_ - q) / delta
            assert torch.allclose(J_hat[:, i, j], J, atol=1e-4)


def test_T_inv_wrt_T(delta: float = 1e-6):
    x = get_random_transform()
    J_hat = torram_jacobians.T_inv_wrt_T(x).view(16, 4, 4)
    x_inv = torram.geometry.inverse_transformation(x)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[i, j] += delta
            x_inv_ = torram.geometry.inverse_transformation(x_)
            J = (x_inv_ - x_inv) / delta
            J = torch.flatten(J)
            assert torch.allclose(J, J_hat[:, i, j])
