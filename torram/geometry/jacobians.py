import torch


__all__ = ['T_wrt_t',
           't_wrt_T',
           'T_wrt_q4d',
           'q4d_wrt_T',
           'T_inv_wrt_T']


def T_wrt_t(t: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((16, 3), dtype=t.dtype, device=t.device)
    J[3, 0] = 1
    J[7, 1] = 1
    J[11, 2] = 1
    return J


def T_wrt_q4d(q4d: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((4, 4, 4), dtype=q4d.dtype, device=q4d.device)
    qx, qy, qz, qw = q4d

    J[:3, :3, 0] = torch.tensor(
        [[4.0*qx*(qy**2 + qz**2),
          4.0*qx*(qw*qz - qx*qy) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qx*(qw*qy + qx*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-4.0*qx*(qw*qz + qx*qy) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qx*(-qw**2 - qy**2),
          -2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qx*(qw*qx - qy*qz)],
         [4.0*qx*(qw*qy - qx*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qx*(qw*qx + qy*qz),
          4.0*qx*(-qw**2 - qz**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[:3, :3, 1] = torch.tensor(
        [[4.0*qy*(-qw**2 - qx**2),
          2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qy*(qw*qz - qx*qy),
          2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qy*(qw*qy + qx*qz)],
         [2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qy*(qw*qz + qx*qy),
          4.0*qy*(qx**2 + qz**2),
          4.0*qy*(qw*qx - qy*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qy*(qw*qy - qx*qz),
          -4.0*qy*(qw*qx + qy*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qy*(-qw**2 - qz**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[:3, :3, 2] = torch.tensor(
        [[4.0*qz*(-qw**2 - qx**2),
          -2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qz - qx*qy),
          2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qy + qx*qz)],
         [2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qz + qx*qy),
          4.0*qz*(-qw**2 - qy**2),
          2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qx - qy*qz)],
         [2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qy - qx*qz),
          2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qx + qy*qz),
          4.0*qz*(qx**2 + qy**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[:3, :3, 3] = torch.tensor(
        [[4.0*qw*(qy**2 + qz**2),
          4.0*qw*(qw*qz - qx*qy) - 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qw*(qw*qy + qx*qz) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-4.0*qw*(qw*qz + qx*qy) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qw*(qx**2 + qz**2),
          4.0*qw*(qw*qx - qy*qz) - 2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2)],
         [4.0*qw*(qw*qy - qx*qz) - 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qw*(qw*qx + qy*qz) + 2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qw*(qx**2 + qy**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J = torch.flatten(J, end_dim=-2)
    return J


def q4d_wrt_T(T: torch.Tensor) -> torch.Tensor:
    # m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(T[..., :3, :3], chunks=9, dim=-1)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.flatten(T[:3, :3])

    J_trace_positive_cond = torch.zeros((4, 4, 4), dtype=T.dtype, device=T.device)
    J_trace_positive_cond[0, 0, 0] = 0.25*(m12 - m21)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[0, 0, 1] = 0.25*(-m02 + m20)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[0, 0, 2] = 0.25*(m01 - m10)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[0, 0, 3] = 0.25/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[0, 1, 0] = 0
    J_trace_positive_cond[0, 1, 1] = 0
    J_trace_positive_cond[0, 1, 2] = -0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[0, 1, 3] = 0
    J_trace_positive_cond[0, 2, 0] = 0
    J_trace_positive_cond[0, 2, 1] = 0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[0, 2, 2] = 0
    J_trace_positive_cond[0, 2, 3] = 0
    J_trace_positive_cond[1, 0, 0] = 0
    J_trace_positive_cond[1, 0, 1] = 0
    J_trace_positive_cond[1, 0, 2] = 0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[1, 0, 3] = 0
    J_trace_positive_cond[1, 1, 0] = 0.25*(m12 - m21)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[1, 1, 1] = 0.25*(-m02 + m20)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[1, 1, 2] = 0.25*(m01 - m10)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[1, 1, 3] = 0.25/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[1, 2, 0] = -0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[1, 2, 1] = 0
    J_trace_positive_cond[1, 2, 2] = 0
    J_trace_positive_cond[1, 2, 3] = 0
    J_trace_positive_cond[2, 0, 0] = 0
    J_trace_positive_cond[2, 0, 1] = -0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[2, 0, 2] = 0
    J_trace_positive_cond[2, 0, 3] = 0
    J_trace_positive_cond[2, 1, 0] = 0.5/torch.sqrt(m00 + m11 + m22 + 1.0)
    J_trace_positive_cond[2, 1, 1] = 0
    J_trace_positive_cond[2, 1, 2] = 0
    J_trace_positive_cond[2, 1, 3] = 0
    J_trace_positive_cond[2, 2, 0] = 0.25*(m12 - m21)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[2, 2, 1] = 0.25*(-m02 + m20)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[2, 2, 2] = 0.25*(m01 - m10)/(m00 + m11 + m22 + 1.0)**(3/2)
    J_trace_positive_cond[2, 2, 3] = 0.25/torch.sqrt(m00 + m11 + m22 + 1.0)

    J_cond1 = torch.zeros((4, 4, 4), dtype=T.dtype, device=T.device)
    J_cond1[0, 0, 0] = 0.25/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[0, 0, 1] = 0.25*(-m01 - m10)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[0, 0, 2] = 0.25*(-m02 - m20)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[0, 0, 3] = 0.25*(m12 - m21)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[0, 1, 0] = 0
    J_cond1[0, 1, 1] = 0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[0, 1, 2] = 0
    J_cond1[0, 1, 3] = 0
    J_cond1[0, 2, 0] = 0
    J_cond1[0, 2, 1] = 0
    J_cond1[0, 2, 2] = 0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[0, 2, 3] = 0
    J_cond1[1, 0, 0] = 0
    J_cond1[1, 0, 1] = 0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[1, 0, 2] = 0
    J_cond1[1, 0, 3] = 0
    J_cond1[1, 1, 0] = -0.25/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[1, 1, 1] = 0.25*(m01 + m10)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[1, 1, 2] = 0.25*(m02 + m20)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[1, 1, 3] = 0.25*(-m12 + m21)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[1, 2, 0] = 0
    J_cond1[1, 2, 1] = 0
    J_cond1[1, 2, 2] = 0
    J_cond1[1, 2, 3] = -0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[2, 0, 0] = 0
    J_cond1[2, 0, 1] = 0
    J_cond1[2, 0, 2] = 0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[2, 0, 3] = 0
    J_cond1[2, 1, 0] = 0
    J_cond1[2, 1, 1] = 0
    J_cond1[2, 1, 2] = 0
    J_cond1[2, 1, 3] = 0.5/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[2, 2, 0] = -0.25/torch.sqrt(m00 - m11 - m22 + 1.0)
    J_cond1[2, 2, 1] = 0.25*(m01 + m10)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[2, 2, 2] = 0.25*(m02 + m20)/(m00 - m11 - m22 + 1.0)**(3/2)
    J_cond1[2, 2, 3] = 0.25*(-m12 + m21)/(m00 - m11 - m22 + 1.0)**(3/2)

    J_cond2 = torch.zeros((4, 4, 4), dtype=T.dtype, device=T.device)
    J_cond2[0, 0, 0] = 0.25*(m01 + m10)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[0, 0, 1] = -0.25/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[0, 0, 2] = 0.25*(m12 + m21)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[0, 0, 3] = 0.25*(m02 - m20)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[0, 1, 0] = 0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[0, 1, 1] = 0
    J_cond2[0, 1, 2] = 0
    J_cond2[0, 1, 3] = 0
    J_cond2[0, 2, 0] = 0
    J_cond2[0, 2, 1] = 0
    J_cond2[0, 2, 2] = 0
    J_cond2[0, 2, 3] = 0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[1, 0, 0] = 0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[1, 0, 1] = 0
    J_cond2[1, 0, 2] = 0
    J_cond2[1, 0, 3] = 0
    J_cond2[1, 1, 0] = 0.25*(-m01 - m10)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[1, 1, 1] = 0.25/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[1, 1, 2] = 0.25*(-m12 - m21)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[1, 1, 3] = 0.25*(-m02 + m20)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[1, 2, 0] = 0
    J_cond2[1, 2, 1] = 0
    J_cond2[1, 2, 2] = 0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[1, 2, 3] = 0
    J_cond2[2, 0, 0] = 0
    J_cond2[2, 0, 1] = 0
    J_cond2[2, 0, 2] = 0
    J_cond2[2, 0, 3] = -0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[2, 1, 0] = 0
    J_cond2[2, 1, 1] = 0
    J_cond2[2, 1, 2] = 0.5/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[2, 1, 3] = 0
    J_cond2[2, 2, 0] = 0.25*(m01 + m10)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[2, 2, 1] = -0.25/torch.sqrt(-m00 + m11 - m22 + 1.0)
    J_cond2[2, 2, 2] = 0.25*(m12 + m21)/(-m00 + m11 - m22 + 1.0)**(3/2)
    J_cond2[2, 2, 3] = 0.25*(m02 - m20)/(-m00 + m11 - m22 + 1.0)**(3/2)

    J_cond3 = torch.zeros((4, 4, 4), dtype=T.dtype, device=T.device)
    J_cond3[0, 0, 0] = 0.25*(m02 + m20)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[0, 0, 1] = 0.25*(m12 + m21)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[0, 0, 2] = -0.25/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[0, 0, 3] = 0.25*(-m01 + m10)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[0, 1, 0] = 0
    J_cond3[0, 1, 1] = 0
    J_cond3[0, 1, 2] = 0
    J_cond3[0, 1, 3] = -0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[0, 2, 0] = 0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[0, 2, 1] = 0
    J_cond3[0, 2, 2] = 0
    J_cond3[0, 2, 3] = 0
    J_cond3[1, 0, 0] = 0
    J_cond3[1, 0, 1] = 0
    J_cond3[1, 0, 2] = 0
    J_cond3[1, 0, 3] = 0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[1, 1, 0] = 0.25*(m02 + m20)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[1, 1, 1] = 0.25*(m12 + m21)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[1, 1, 2] = -0.25/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[1, 1, 3] = 0.25*(-m01 + m10)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[1, 2, 0] = 0
    J_cond3[1, 2, 1] = 0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[1, 2, 2] = 0
    J_cond3[1, 2, 3] = 0
    J_cond3[2, 0, 0] = 0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[2, 0, 1] = 0
    J_cond3[2, 0, 2] = 0
    J_cond3[2, 0, 3] = 0
    J_cond3[2, 1, 0] = 0
    J_cond3[2, 1, 1] = 0.5/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[2, 1, 2] = 0
    J_cond3[2, 1, 3] = 0
    J_cond3[2, 2, 0] = 0.25*(-m02 - m20)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[2, 2, 1] = 0.25*(-m12 - m21)/(-m00 - m11 + m22 + 1.0)**(3/2)
    J_cond3[2, 2, 2] = 0.25/torch.sqrt(-m00 - m11 + m22 + 1.0)
    J_cond3[2, 2, 3] = 0.25*(m01 - m10)/(-m00 - m11 + m22 + 1.0)**(3/2)

    trace = m00 + m11 + m22
    where_2 = torch.where(m11 > m22, J_cond2, J_cond3)
    where_1 = torch.where((m00 > m11) & (m00 > m22), J_cond1, where_2)
    Jq = torch.where(trace > 0.0, J_trace_positive_cond, where_1)
    Jq = torch.permute(Jq, (2, 0, 1))
    return torch.flatten(Jq, start_dim=1)


def t_wrt_T(T: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((3, 16), dtype=T.dtype, device=T.device)
    J[0, 3] = 1
    J[1, 7] = 1
    J[2, 11] = 1
    return J


def T_inv_wrt_T(T: torch.Tensor) -> torch.Tensor:
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = torch.flatten(T[:3, :3])
    tx, ty, tz = T[:3, 3]

    J = torch.zeros((4, 4, 4, 4), dtype=T.dtype, device=T.device)
    J[0, 0, 0, 0] = 1  # wrt r00
    J[0, 3, 0, 0] = -tx

    J[1, 0, 0, 1] = 1  # wrt r01
    J[1, 3, 0, 1] = -tx

    J[2, 0, 0, 2] = 1  # wrt r02
    J[2, 3, 0, 2] = -tx

    J[0, 3, 0, 3] = -r00  # wrt tx
    J[1, 3, 0, 3] = -r01
    J[2, 3, 0, 3] = -r02

    J[0, 1, 1, 0] = 1  # wrt r10
    J[0, 3, 1, 0] = -ty

    J[1, 1, 1, 1] = 1  # wrt r11
    J[1, 3, 1, 1] = -ty

    J[2, 1, 1, 2] = 1  # wrt r12
    J[2, 3, 1, 2] = -ty

    J[0, 3, 1, 3] = -r10  # wrt ty
    J[1, 3, 1, 3] = -r11
    J[2, 3, 1, 3] = -r12

    J[0, 2, 2, 0] = 1  # wrt r20
    J[0, 3, 2, 0] = -tz

    J[1, 2, 2, 1] = 1  # wrt r21
    J[1, 3, 2, 1] = -tz

    J[2, 2, 2, 2] = 1  # wrt r22
    J[2, 3, 2, 2] = -tz

    J[0, 3, 2, 3] = -r20  # wrt tz
    J[1, 3, 2, 3] = -r21
    J[2, 3, 2, 3] = -r22

    return J.view(16, 16)
