import torch


__all__ = ['T_tq4d',
           'tq4d_T',
           'static_transform']


def T_tq4d(t: torch.Tensor, q4d: torch.Tensor) -> torch.Tensor:
    J = torch.zeros((7, 4, 4), dtype=t.dtype, device=t.device)
    qx, qy, qz, qw = q4d

    J[0, 0, 3] = 1
    J[1, 1, 3] = 1
    J[2, 2, 3] = 1

    J[3, :3, :3] = torch.tensor(
        [[4.0*qx*(qy**2 + qz**2),
          4.0*qx*(qw*qz - qx*qy) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qx*(qw*qy + qx*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-4.0*qx*(qw*qz + qx*qy) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qx*(-qw**2 - qy**2),
          -2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qx*(qw*qx - qy*qz)],
         [4.0*qx*(qw*qy - qx*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qx*(qw*qx + qy*qz),
          4.0*qx*(-qw**2 - qz**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[4, :3, :3] = torch.tensor(
        [[4.0*qy*(-qw**2 - qx**2),
          2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qy*(qw*qz - qx*qy),
          2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qy*(qw*qy + qx*qz)],
         [2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qy*(qw*qz + qx*qy),
          4.0*qy*(qx**2 + qz**2),
          4.0*qy*(qw*qx - qy*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qy*(qw*qy - qx*qz),
          -4.0*qy*(qw*qx + qy*qz) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qy*(-qw**2 - qz**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[5, :3, :3] = torch.tensor(
        [[4.0*qz*(-qw**2 - qx**2),
          -2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qz - qx*qy),
          2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qy + qx*qz)],
         [2.0*qw*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qz + qx*qy),
          4.0*qz*(-qw**2 - qy**2),
          2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qx - qy*qz)],
         [2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2) + 4.0*qz*(qw*qy - qx*qz),
          2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2) - 4.0*qz*(qw*qx + qy*qz),
          4.0*qz*(qx**2 + qy**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    J[6, :3, :3] = torch.tensor(
        [[4.0*qw*(qy**2 + qz**2),
          4.0*qw*(qw*qz - qx*qy) - 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qw*(qw*qy + qx*qz) + 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2)],
         [-4.0*qw*(qw*qz + qx*qy) + 2.0*qz*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qw*(qx**2 + qz**2),
          4.0*qw*(qw*qx - qy*qz) - 2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2)],
         [4.0*qw*(qw*qy - qx*qz) - 2.0*qy*(qw**2 + qx**2 + qy**2 + qz**2),
          -4.0*qw*(qw*qx + qy*qz) + 2.0*qx*(qw**2 + qx**2 + qy**2 + qz**2),
          4.0*qw*(qx**2 + qy**2)]]) / (qw**2 + qx**2 + qy**2 + qz**2)**2

    return J


def tq4d_T(T: torch.Tensor) -> torch.Tensor:
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
    Jt = torch.zeros((3, 4, 4), dtype=T.dtype, device=T.device)
    Jt[0, 0, 3] = 1
    Jt[1, 1, 3] = 1
    Jt[2, 2, 3] = 1
    return torch.cat([Jt, Jq], dim=0)


def static_transform(sigmas: torch.Tensor, transform: torch.Tensor):
    return torch.einsum('...ij, ...ij, ...ji->...ij', transform, sigmas, transform)
