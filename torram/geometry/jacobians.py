import torch


__all__ = ['T_wrt_t',
           't_wrt_T',
           'T_wrt_q4d',
           'q4d_wrt_T',
           'T_inv_wrt_T']


def T_wrt_t(t: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((*t.shape[:-1], 16, 3), dtype=t.dtype, device=t.device)
    J[..., 3, 0] = 1
    J[..., 7, 1] = 1
    J[..., 11, 2] = 1
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

    J = J / (norm**2)[..., None, None, None]
    J = torch.flatten(J, start_dim=-3, end_dim=-2)
    return J


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
    return torch.flatten(J[..., [1, 2, 3, 0], :, :], start_dim=-2)


def t_wrt_T(T: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((*T.shape[:-2], 3, 16), dtype=T.dtype, device=T.device)
    J[..., 0, 3] = 1
    J[..., 1, 7] = 1
    J[..., 2, 11] = 1
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

    return J.view(*T.shape[:-2], 16, 16)
