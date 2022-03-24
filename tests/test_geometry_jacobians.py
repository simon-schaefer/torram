import kornia
import torch

from torram.geometry import pose_to_transformation_matrix
from torram.geometry import jacobians as torram_jacobians


def get_random_transform():
    t = torch.rand(3, dtype=torch.float64)
    q = torch.rand(4, dtype=torch.float64)
    return pose_to_transformation_matrix(t, q)


def test_jacobian_transform_tq4d(delta: float = 1e-6):
    x = torch.rand(3 + 4, dtype=torch.float64)
    J_hat = torram_jacobians.T_tq4d(x[:3], x[3:])
    T = pose_to_transformation_matrix(x[:3], x[3:])
    for k in range(3 + 4):
        x_ = x.clone()
        x_[k] += delta
        Td = pose_to_transformation_matrix(x_[:3], x_[3:])
        J = (Td - T) / delta
        assert torch.allclose(J, J_hat[k], atol=1e-4)


def test_jacobian_tq4d_transform(delta: float = 1e-6):
    x = get_random_transform()
    xq = kornia.geometry.rotation_matrix_to_quaternion(x[:3, :3].contiguous())

    J_hat = torram_jacobians.tq4d_T(x)
    for i in range(4):
        for j in range(4):
            x_ = x.clone()
            x_[i, j] += delta
            q = kornia.geometry.rotation_matrix_to_quaternion(x_[:3, :3].contiguous())
            Jt = (x_[:3, 3] - x[:3, 3]) / delta
            Jq = (q - xq) / delta
            assert torch.allclose(Jt, J_hat[:3, i, j])
            assert torch.allclose(Jq, J_hat[3:, i, j])
