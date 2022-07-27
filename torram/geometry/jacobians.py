import math
import torch

from torch.distributions import Normal, MultivariateNormal
from torram.geometry import diag_last
from typing import Union

__all__ = ['T_wrt_t',
           't_wrt_T',
           'T_wrt_q3d',
           'T_wrt_q4d',
           'q3d_wrt_T',
           'q4d_wrt_T',
           'T_inv_wrt_T',
           'cov_error_propagation']


def T_wrt_t(t: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((*t.shape[:-1], 4, 4, 3), dtype=t.dtype, device=t.device)
    J[..., 0, 3, 0] = 1
    J[..., 1, 3, 1] = 1
    J[..., 2, 3, 2] = 1
    return J


def q3d_wrt_T(T: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    R_flat = torch.flatten(T[..., :3, :3], start_dim=-2)
    chunks = torch.chunk(R_flat, chunks=9, dim=-1)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = (c[..., 0] for c in chunks)
    batch_size = T.shape[:-2]

    # Singularity => No rotation when rotation matrix is an identity matrix. Then the angle axis reprensentation
    # should be zeros, as they have been initialized.
    J_sx = torch.zeros((*batch_size, 3, 4, 4), dtype=T.dtype, device=T.device)
    J_sx[..., 0, 0, 0] = 1
    J_sx[..., 1, 0, 0] = (r01 - r10) / (2 * (r00 + 1))
    J_sx[..., 2, 0, 0] = (r20 - r02) / (2 * (r00 + 1))
    J_sx[..., 1, 0, 1] = -1
    J_sx[..., 2, 0, 2] = 1
    J_sx[..., 1, 1, 0] = 1
    J_sx[..., 2, 2, 0] = -1
    factor_x = (math.sqrt(2) * math.pi) / (4 * torch.sqrt(r00 + 1))
    J_sx = J_sx * factor_x.view(*batch_size, 1, 1, 1)

    J_sy = torch.zeros((*batch_size, 3, 4, 4), dtype=T.dtype, device=T.device)
    J_sy[..., 0, 0, 1] = -1
    J_sy[..., 0, 1, 0] = 1
    J_sy[..., 0, 1, 1] = (r01 - r10) / (2 * (r11 + 1))
    J_sy[..., 1, 1, 1] = 1
    J_sy[..., 2, 1, 1] = (r12 - r21) / (2 * (r11 + 1))
    J_sy[..., 2, 1, 2] = -1
    J_sy[..., 2, 2, 1] = 1
    factor_y = (math.sqrt(2) * math.pi) / (4 * torch.sqrt(r11 + 1))
    J_sy = J_sy * factor_y.view(*batch_size, 1, 1, 1)

    J_sz = torch.zeros((*batch_size, 3, 4, 4), dtype=T.dtype, device=T.device)
    J_sz[..., 0, 0, 2] = 1
    J_sz[..., 1, 1, 2] = -1
    J_sz[..., 0, 0, 2] = -1
    J_sz[..., 1, 2, 1] = 1
    J_sz[..., 1, 2, 1] = 1
    J_sz[..., 0, 2, 2] = (r20 - r02) / (2 * (r22 + 1))
    J_sz[..., 1, 2, 2] = (r12 - r21) / (2 * (r22 + 1))
    J_sz[..., 2, 2, 2] = 1
    factor_z = (math.sqrt(2) * math.pi) / (4 * torch.sqrt(r22 + 1))
    J_sz = J_sz * factor_z.view(*batch_size, 1, 1, 1)

    # "Normal" transformation jacobian.
    a = torch.clamp(r00 + r11 + r22 - 1, -2 + epsilon, 2 - epsilon)
    b = torch.sqrt((r01 - r10)**2 + (r02 - r20)**2 + (r12 - r21)**2) + epsilon
    c = torch.acos(a / 2)
    norm1 = torch.sqrt(4 - a**2) * b
    norm2 = b ** 3 + epsilon
    J_n = torch.zeros((*batch_size, 3, 4, 4), dtype=T.dtype, device=T.device)
    J_n[..., 0, 0, 0] = (r12 - r21) / norm1
    J_n[..., 1, 0, 0] = (r20 - r02) / norm1
    J_n[..., 2, 0, 0] = (r01 - r10) / norm1
    J_n[..., 0, 0, 1] = (r01 - r10) * (r12 - r21) * c / norm2
    J_n[..., 1, 0, 1] = -(r01 - r10) * (r02 - r20) * c / norm2
    J_n[..., 2, 0, 1] = -((r20 - r02)**2 + (r12 - r21)**2) * c / norm2
    J_n[..., 0, 0, 2] = (r02 - r20) * (r12 - r21) * c / norm2
    J_n[..., 1, 0, 2] = ((r01 - r10)**2 + (r12 - r21)**2) * c / norm2
    J_n[..., 2, 0, 2] = (r01 - r10) * (r02 - r20) * c / norm2
    J_n[..., 0, 1, 0] = -(r01 - r10) * (r12 - r21) * c / norm2
    J_n[..., 1, 1, 0] = (r01 - r10) * (r02 - r20) * c / norm2
    J_n[..., 2, 1, 0] = ((r02 - r20)**2 + (r12 - r21)**2) * c / norm2
    J_n[..., 0, 1, 1] = (r12 - r21) / norm1
    J_n[..., 1, 1, 1] = (r20 - r02) / norm1
    J_n[..., 2, 1, 1] = (r01 - r10) / norm1
    J_n[..., 0, 1, 2] = - ((r01 - r10)**2 + (r20 - r02)**2) * c / norm2
    J_n[..., 1, 1, 2] = -(r02 - r20) * (r12 - r21) * c / norm2
    J_n[..., 2, 1, 2] = (r01 - r10) * (r12 - r21) * c / norm2
    J_n[..., 0, 2, 0] = -(r02 - r20) * (r12 - r21) * c / norm2
    J_n[..., 1, 2, 0] = - ((r01 - r10)**2 + (r12 - r21)**2) * c / norm2
    J_n[..., 2, 2, 0] = -(r01 - r10) * (r02 - r20) * c / norm2
    J_n[..., 0, 2, 1] = ((r01 - r10)**2 + (r02 - r20)**2) * c / norm2
    J_n[..., 1, 2, 1] = (r02 - r20) * (r12 - r21) * c / norm2
    J_n[..., 2, 2, 1] = -(r01 - r10) * (r12 - r21) * c / norm2
    J_n[..., 0, 2, 2] = (r12 - r21) / norm1
    J_n[..., 1, 2, 2] = (r20 - r02) / norm1
    J_n[..., 2, 2, 2] = (r01 - r10) / norm1

    # Conditional output.
    is_singular = ((r21 - r12).abs() < epsilon) & ((r02 - r20).abs() < epsilon) & ((r10 - r01).abs() < epsilon)
    is_singular = is_singular.view(*batch_size, 1, 1, 1)
    is_no_rotation = ((r21 + r12).abs() < epsilon) & ((r02 + r20).abs() < epsilon) & ((r10 + r01).abs() < epsilon) \
                     & ((r00 + r11 + r22 - 3).abs() < epsilon)
    is_no_rotation = is_no_rotation.view(*batch_size, 1, 1, 1)
    is_sing_rot = is_singular & ~is_no_rotation

    xx = (r00.view(*batch_size, 1, 1, 1) + 1) / 2
    yy = (r11.view(*batch_size, 1, 1, 1) + 1) / 2
    zz = (r22.view(*batch_size, 1, 1, 1) + 1) / 2

    J = torch.zeros((*batch_size, 3, 4, 4), dtype=T.dtype, device=T.device)
    J = torch.where(is_sing_rot & (xx > yy) & (xx > zz) & (xx >= epsilon), J_sx, J)
    J = torch.where(is_sing_rot & (yy > xx) & (yy > zz) & (yy >= epsilon), J_sy, J)
    J = torch.where(is_sing_rot & (zz > xx) & (zz > yy) & (zz >= epsilon), J_sz, J)
    return torch.where(~is_singular, J_n, J)


def T_wrt_q3d(q3d: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Jacobian of the transformation q3d (axis-angle vector) -> T (transformation matrix).
    Math inspired by: https://github.com/dorianhenning/ba-srl/blob/smpl-refactor/scripts/compute_smpl_jacobian.py
    """
    batch_size = q3d.shape[:-1]
    J = torch.zeros((*batch_size, 4, 4, 3), dtype=q3d.dtype, device=q3d.device)

    phi = torch.norm(q3d, dim=-1, keepdim=True) + eps
    phi2 = phi**2
    sin_t = torch.sin(phi)
    sinc_t = sin_t / phi
    cos_t = torch.cos(phi)
    c_1 = cos_t - 1.0

    K_ = __batch_skew(q3d)
    K_2 = K_ @ K_
    K_ = torch.flatten(K_, start_dim=-2)
    K_2 = torch.flatten(K_2, start_dim=-2)

    x_vec = torch.tensor([[1, 0, 0]], dtype=q3d.dtype, device=q3d.device)
    y_vec = torch.tensor([[0, 1, 0]], dtype=q3d.dtype, device=q3d.device)
    z_vec = torch.tensor([[0, 0, 1]], dtype=q3d.dtype, device=q3d.device)
    generator_Rx = __batch_skew(x_vec).reshape(9, 1)
    generator_Ry = __batch_skew(y_vec).reshape(9, 1)
    generator_Rz = __batch_skew(z_vec).reshape(9, 1)

    rx = q3d[..., 0].unsqueeze(-1)
    ry = q3d[..., 1].unsqueeze(-1)
    rz = q3d[..., 2].unsqueeze(-1)
    zeros = torch.zeros_like(rx)

    special_Kx = torch.cat([zeros, - ry, - rz, - ry, 2 * rx, zeros, - rz, zeros, 2 * rx], dim=-1).unsqueeze(-1)
    special_Ky = torch.cat([2 * ry, - rx, zeros, - rx, zeros, - rz, zeros, - rz, 2 * ry], dim=-1).unsqueeze(-1)
    special_Kz = torch.cat([2 * rz, zeros, - rx, zeros, 2 * rz, - ry, - rx, - ry, zeros], dim=-1).unsqueeze(-1)

    special_K = torch.cat((special_Kx, special_Ky, special_Kz), dim=-1)
    special_K = torch.einsum('...i,...jk->...jk', c_1 / phi2, special_K)

    generator = torch.cat((generator_Rx, generator_Ry, generator_Rz), dim=-1)
    generator = torch.einsum('...i,jk->...jk', sinc_t, generator)

    M = 2 * c_1 / phi2**2 * K_2 + sinc_t / phi2 * (K_2 - K_) + cos_t / phi2 * K_
    M_rotvec = torch.einsum('...i,...j->...ij', M, q3d)
    J_rot = M_rotvec + special_K + generator
    J[..., :3, :3, :] = J_rot.view(*batch_size, 3, 3, 3)
    return J


def T_wrt_q4d(q4d: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((*q4d.shape[:-1], 4, 4, 4), dtype=q4d.dtype, device=q4d.device)
    qx, qy, qz, qw = torch.chunk(q4d, chunks=4, dim=-1)
    qx = qx[..., 0]
    qy = qy[..., 0]
    qz = qz[..., 0]
    qw = qw[..., 0]

    norm = (qw**2 + qx**2 + qy**2 + qz**2)
    wz_minus_xy = (qw*qz - qx*qy)
    wy_plus_xz = (qw*qy + qx*qz)
    wz_plus_xy = (qw*qz + qx*qy)
    wx_minus_yz = (qw*qx - qy*qz)
    wy_minus_xz = (qw*qy - qx*qz)
    wx_plus_yz = (qw*qx + qy*qz)

    J[..., 0, 0, 0] = (4.0*qx*(qy**2 + qz**2))
    J[..., 0, 1, 0] = (4.0*qx*wz_minus_xy + 2.0*qy*norm)
    J[..., 0, 2, 0] = (-4.0*qx*wy_plus_xz + 2.0*qz*norm)
    J[..., 1, 0, 0] = (-4.0*qx*wz_plus_xy + 2.0*qy*norm)
    J[..., 1, 1, 0] = (4.0*qx*(-qw**2 - qy**2))
    J[..., 1, 2, 0] = (-2.0*qw*norm + 4.0*qx*wx_minus_yz)
    J[..., 2, 0, 0] = (4.0*qx*wy_minus_xz + 2.0*qz*norm)
    J[..., 2, 1, 0] = (2.0*qw*norm - 4.0*qx*wx_plus_yz)
    J[..., 2, 2, 0] = (4.0*qx*(-qw**2 - qz**2))

    J[..., 0, 0, 1] = (4.0*qy*(-qw**2 - qx**2))
    J[..., 0, 1, 1] = (2.0*qx*norm + 4.0*qy*wz_minus_xy)
    J[..., 0, 2, 1] = (2.0*qw*norm - 4.0*qy*wy_plus_xz)
    J[..., 1, 0, 1] = (2.0*qx*norm - 4.0*qy*wz_plus_xy)
    J[..., 1, 1, 1] = (4.0*qy*(qx**2 + qz**2))
    J[..., 1, 2, 1] = (4.0*qy*wx_minus_yz + 2.0*qz*norm)
    J[..., 2, 0, 1] = (-2.0*qw*norm + 4.0*qy*wy_minus_xz)
    J[..., 2, 1, 1] = (-4.0*qy*wx_plus_yz + 2.0*qz*norm)
    J[..., 2, 2, 1] = (4.0*qy*(-qw**2 - qz**2))

    J[..., 0, 0, 2] = (4.0*qz*(-qw**2 - qx**2))
    J[..., 0, 1, 2] = (-2.0*qw*norm + 4.0*qz*wz_minus_xy)
    J[..., 0, 2, 2] = (2.0*qx*norm - 4.0*qz*wy_plus_xz)
    J[..., 1, 0, 2] = (2.0*qw*norm - 4.0*qz*wz_plus_xy)
    J[..., 1, 1, 2] = (4.0*qz*(-qw**2 - qy**2))
    J[..., 1, 2, 2] = (2.0*qy*norm + 4.0*qz*wx_minus_yz)
    J[..., 2, 0, 2] = (2.0*qx*norm + 4.0*qz*wy_minus_xz)
    J[..., 2, 1, 2] = (2.0*qy*norm - 4.0*qz*wx_plus_yz)
    J[..., 2, 2, 2] = (4.0*qz*(qx**2 + qy**2))

    J[..., 0, 0, 3] = (4.0*qw*(qy**2 + qz**2))
    J[..., 0, 1, 3] = (4.0*qw*wz_minus_xy - 2.0*qz*norm)
    J[..., 0, 2, 3] = (-4.0*qw*wy_plus_xz + 2.0*qy*norm)
    J[..., 1, 0, 3] = (-4.0*qw*wz_plus_xy + 2.0*qz*norm)
    J[..., 1, 1, 3] = (4.0*qw*(qx**2 + qz**2))
    J[..., 1, 2, 3] = (4.0*qw*wx_minus_yz - 2.0*qx*norm)
    J[..., 2, 0, 3] = (4.0*qw*wy_minus_xz - 2.0*qy*norm)
    J[..., 2, 1, 3] = (-4.0*qw*wx_plus_yz + 2.0*qx*norm)
    J[..., 2, 2, 3] = (4.0*qw*(qx**2 + qy**2))

    return J / (norm**2)[..., None, None, None]


def q4d_wrt_T(T: torch.Tensor) -> torch.Tensor:
    shape = T.shape[:-2]
    T = T.view(-1, 4, 4)
    batch_size = T.shape[0]

    J = torch.zeros((batch_size, 4, 4, 4), dtype=T.dtype, device=T.device)
    R = T[:, :3, :3]
    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    tgz = t > 0
    tlz = t <= 0
    eye = torch.eye(3, dtype=T.dtype, device=T.device)

    if torch.any(tgz):
        isqrt_t = 1 / torch.sqrt(1 + t[tgz])
        J[tgz, 0, :3, :3] = 0.25 * isqrt_t[..., None, None] * eye
        J[tgz, 1, :3, :3] = -0.25 * (isqrt_t / (1 + t[tgz]) * (R[tgz, 2, 1] - R[tgz, 1, 2]))[..., None, None] * eye
        J[tgz, 1, 2, 1] = 0.5 * isqrt_t
        J[tgz, 1, 1, 2] = -0.5 * isqrt_t
        J[tgz, 2, :3, :3] = -0.25 * (isqrt_t / (1 + t[tgz]) * (R[tgz, 0, 2] - R[tgz, 2, 0]))[..., None, None] * eye
        J[tgz, 2, 0, 2] = 0.5 * isqrt_t
        J[tgz, 2, 2, 0] = -0.5 * isqrt_t
        J[tgz, 3, :3, :3] = -0.25 * (isqrt_t / (1 + t[tgz]) * (R[tgz, 1, 0] - R[tgz, 0, 1]))[..., None, None] * eye
        J[tgz, 3, 1, 0] = 0.5 * isqrt_t
        J[tgz, 3, 0, 1] = -0.5 * isqrt_t

    if torch.any(tlz):
        batch_size_tlz = torch.count_nonzero(tlz)
        batch_index_tlz = list(range(batch_size_tlz))
        i = torch.argmax(torch.stack([R[tlz, 0, 0], R[tlz, 1, 1], R[tlz, 2, 2]], dim=-1), dim=-1)
        j = (i + 1) % 3
        k = (j + 1) % 3

        r = torch.sqrt(R[tlz, i, i] - R[tlz, j, j] - R[tlz, k, k] + 1)
        i_r = 1 / r
        i_r_cube = 1 / ((R[tlz, i, i] - R[tlz, j, j] - R[tlz, k, k] + 1) * r)
        r_eye = eye[None].repeat(batch_size_tlz, 1, 1)
        r_eye[batch_index_tlz, j, j] = -1
        r_eye[batch_index_tlz, k, k] = -1
        J[tlz, 1 + i, :3, :3] = 0.25 * i_r[..., None, None] * r_eye
        J[tlz, 0, :3, :3] = -0.25 * ((R[tlz, k, j] - R[tlz, j, k]) * i_r_cube)[:, None, None] * r_eye
        J[tlz, 0, k, j] = 0.5 * i_r
        J[tlz, 0, j, k] = -0.5 * i_r
        J[tlz, 1 + j, :3, :3] = -0.25 * ((R[tlz, j, i] + R[tlz, i, j]) * i_r_cube)[:, None, None] * r_eye  # noqa
        J[tlz, 1 + j, j, i] = 0.5 * i_r
        J[tlz, 1 + j, i, j] = 0.5 * i_r
        J[tlz, 1 + k, :3, :3] = -0.25 * ((R[tlz, k, i] + R[tlz, i, k]) * i_r_cube)[:, None, None] * r_eye  # noqa
        J[tlz, 1 + k, k, i] = 0.5 * i_r
        J[tlz, 1 + k, i, k] = 0.5 * i_r

    J = J.view(*shape, 4, 4, 4)
    return J[..., [1, 2, 3, 0], :, :]


def t_wrt_T(T: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((*T.shape[:-2], 3, 4, 4), dtype=T.dtype, device=T.device)
    J[..., 0, 0, 3] = 1
    J[..., 1, 1, 3] = 1
    J[..., 2, 2, 3] = 1
    return J


def T_inv_wrt_T(T: torch.Tensor) -> torch.Tensor:
    R_flat = torch.flatten(T[..., :3, :3], start_dim=-2)
    chunks = torch.chunk(R_flat, chunks=9, dim=-1)
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = (c[..., 0] for c in chunks)
    chunks = torch.chunk(T[..., :3, 3], chunks=3, dim=-1)
    tx, ty, tz = (c[..., 0] for c in chunks)

    J = torch.zeros((*T.shape[:-2], 4, 4, 4, 4), dtype=T.dtype, device=T.device)
    J[..., 0, 0, 0, 0] = 1  # wrt r00
    J[..., 0, 3, 0, 0] = -tx

    J[..., 1, 0, 0, 1] = 1  # wrt r01
    J[..., 1, 3, 0, 1] = -tx

    J[..., 2, 0, 0, 2] = 1  # wrt r02
    J[..., 2, 3, 0, 2] = -tx

    J[..., 0, 3, 0, 3] = -r00  # wrt tx
    J[..., 1, 3, 0, 3] = -r01
    J[..., 2, 3, 0, 3] = -r02

    J[..., 0, 1, 1, 0] = 1  # wrt r10
    J[..., 0, 3, 1, 0] = -ty

    J[..., 1, 1, 1, 1] = 1  # wrt r11
    J[..., 1, 3, 1, 1] = -ty

    J[..., 2, 1, 1, 2] = 1  # wrt r12
    J[..., 2, 3, 1, 2] = -ty

    J[..., 0, 3, 1, 3] = -r10  # wrt ty
    J[..., 1, 3, 1, 3] = -r11
    J[..., 2, 3, 1, 3] = -r12

    J[..., 0, 2, 2, 0] = 1  # wrt r20
    J[..., 0, 3, 2, 0] = -tz

    J[..., 1, 2, 2, 1] = 1  # wrt r21
    J[..., 1, 3, 2, 1] = -tz

    J[..., 2, 2, 2, 2] = 1  # wrt r22
    J[..., 2, 3, 2, 2] = -tz

    J[..., 0, 3, 2, 3] = -r20  # wrt tz
    J[..., 1, 3, 2, 3] = -r21
    J[..., 2, 3, 2, 3] = -r22

    return J


def cov_error_propagation(
        x: Union[Normal, MultivariateNormal, torch.Tensor],
        Jx: torch.Tensor,
        square_form: bool = False
    ) -> torch.Tensor:
    """Covariance error propagation.

    For a coveriance matrix C which is transformed by some transform T(x) with jacobian J(x) = dT/dx, the
    transformed covariance matrix C' is:

    C' = J * C * J^T

    For numerical stability the square-root form of this equation can be used:

    C* = J * sqrt(C)
    C' = C* * (C*)^T

    Args:
        x: initial distribution. when x is a Normal distribution the covariance matrix will be created as a
           diagonal matrix filled with x's variances.
        Jx: jacobian dT/dx.
        square_form: use the square-root form or the standard form to calculate C'.
    """
    if isinstance(x, Normal):
        x_cov = diag_last(x.variance)
    elif isinstance(x, MultivariateNormal):
        x_cov = x.covariance_matrix
    else:
        x_cov = x

    if square_form:
        cov_ = torch.matmul(Jx, torch.sqrt(x_cov))
        return torch.matmul(cov_, cov_.transpose(-1, -2))
    return torch.einsum('...ij,...jk,...kl->...il', Jx, x_cov, Jx.transpose(-1, -2))


def __batch_skew(vecs, dtype=torch.float64):
    batch_size = vecs.shape[:-1]
    device = vecs.device
    vx, vy, vz = torch.split(vecs, 1, dim=-1)
    zeros = torch.zeros((*batch_size, 1), dtype=dtype, device=device)
    return torch.cat([zeros, -vz, vy, vz, zeros, -vx, -vy, vx, zeros], dim=-1).view((*batch_size, 3, 3))
