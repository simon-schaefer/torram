import kornia
import torch
import kornia.geometry.conversions
from kornia.geometry import (
    convert_points_to_homogeneous,
    transform_points
)
from torch.nn import functional as F
import warnings

__all__ = ['angle_axis_to_quaternion',
           'angle_axis_to_rotation_matrix',
           'angle_axis_to_rotation_6d',
           'quaternion_to_angle_axis',
           'quaternion_to_rotation_matrix',
           'rotation_matrix_to_angle_axis',
           'rotation_matrix_to_quaternion',
           'rotation_matrix_to_rotation_6d',
           'rotation_6d_to_rotation_matrix',
           'rotation_6d_to_axis_angle',
           'rotation_6d_to_quaternion',
           'multiply_angle_axis',
           'multiply_quaternion',
           'inverse_quaternion',
           'inverse_transformation',
           'pose_to_transformation_matrix',
           'transform_points',
           'convert_points_to_homogeneous'
           ]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (x, y, z, w) to a rotation matrix.

    >>> quaternion = torch.tensor((0., 0., 0., 1.))
    >>> quaternion_to_rotation_matrix(quaternion)
    tensor([[-1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.]])

    Args:
        q: a tensor containing a quaternion to be converted (..., 4).
    Return:
        the rotation matrix of shape :math:`(..., 3, 3)`.
    """
    shape = q.shape[:-1]
    if len(shape) == 0:
        q_flat = q[None]
    else:
        q_flat = torch.flatten(q, end_dim=-2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_order = kornia.geometry.conversions.QuaternionCoeffOrder.XYZW
        rotmat_flat = kornia.geometry.quaternion_to_rotation_matrix(q_flat, order=q_order)
    return rotmat_flat.view(*shape, 3, 3)


def quaternion_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector (x, y, z, w) to angle axis of rotation in radians.

    >>> quaternion = torch.tensor((0., 0., 0., 1.))
    >>> quaternion_to_angle_axis(quaternion)
    tensor([3.1416, 0.0000, 0.0000])

    Args:
        q: tensor with quaternions (..., 4).
    Return:
        angle axis rotation vector (..., 3).
    """
    shape = q.shape[:-1]
    if len(shape) == 0:
        q_flat = q[None]
    else:
        q_flat = torch.flatten(q, end_dim=-2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_order = kornia.geometry.conversions.QuaternionCoeffOrder.XYZW
        angle_axis_flat = kornia.geometry.quaternion_to_angle_axis(q_flat, order=q_order)
    return angle_axis_flat.view(*shape, 3)


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert an angle axis to a quaternion (x, y, z, w).

    >>> x = torch.tensor((0., 1., 0.))
    >>> angle_axis_to_quaternion(x)
    tensor([0.0000, 0.4794, 0.0000, 0.8776])

    Args:
        angle_axis: tensor with angle axis in radians (..., 3)
    Return:
        tensor with quaternion (..., 4)
    """
    shape = angle_axis.shape[:-1]
    if len(shape) == 0:
        aa_flat = angle_axis[None]
    else:
        aa_flat = torch.flatten(angle_axis, end_dim=-2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_order = kornia.geometry.conversions.QuaternionCoeffOrder.XYZW
        quaternion_flat = kornia.geometry.angle_axis_to_quaternion(aa_flat, order=q_order)
    return quaternion_flat.view(*shape, 4)


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Convert 3x3 rotation matrix to axis angle representation.
    Math inspired by https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToAngle/index.htm

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(..., 3, 3)`.
        epsilon: norm for singularity check.
    Return:
        the rotation in axis angle representation with shape :math:`(..., 3)`.
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    output = torch.zeros((*rotation_matrix.shape[:-2], 3), dtype=rotation_matrix.dtype, device=rotation_matrix.device)
    trace = (rotation_matrix[..., 0, 0] + rotation_matrix[..., 1, 1] + rotation_matrix[..., 2, 2]).unsqueeze(-1)
    ymz = (rotation_matrix[..., 2, 1] - rotation_matrix[..., 1, 2]).unsqueeze(-1)
    xmz = (rotation_matrix[..., 0, 2] - rotation_matrix[..., 2, 0]).unsqueeze(-1)
    xmy = (rotation_matrix[..., 1, 0] - rotation_matrix[..., 0, 1]).unsqueeze(-1)
    yz = (rotation_matrix[..., 2, 1] + rotation_matrix[..., 1, 2]).unsqueeze(-1)
    xz = (rotation_matrix[..., 0, 2] + rotation_matrix[..., 2, 0]).unsqueeze(-1)
    xy = (rotation_matrix[..., 1, 0] + rotation_matrix[..., 0, 1]).unsqueeze(-1)

    # Singularity handling.
    is_singular = (ymz.abs() < epsilon) & (xmz.abs() < epsilon) & (xmy.abs() < epsilon)
    is_no_rotation = (yz.abs() < epsilon) & (xz.abs() < epsilon) & (xy.abs() < epsilon) & ((trace - 3).abs() < epsilon)
    xx = (rotation_matrix[..., 0, 0] + 1 + epsilon).unsqueeze(-1) / 2
    yy = (rotation_matrix[..., 1, 1] + 1 + epsilon).unsqueeze(-1) / 2
    zz = (rotation_matrix[..., 2, 2] + 1 + epsilon).unsqueeze(-1) / 2
    xy4 = xy / 4
    xz4 = xz / 4
    yz4 = yz / 4

    # 1) Singularity => No rotation when rotation matrix is an identity matrix. Then the angle axis reprensentation
    # should be zeros, as they have been initialized.
    # output[is_singular & is_no_rotation] = 0  # no rotation => angle = 0
    is_sing_rot = is_singular & ~is_no_rotation

    # 2) Singularity => Different variants of 180 degrees rotations.
    x_180 = torch.tensor([torch.pi, 0, 0], dtype=xx.dtype, device=xx.device)
    y_180 = torch.tensor([0, torch.pi, 0], dtype=yy.dtype, device=yy.device)
    z_180 = torch.tensor([0, 0, torch.pi], dtype=zz.dtype, device=zz.device)
    output = torch.where(is_sing_rot & (xx > yy) & (xx > zz) & (xx < epsilon), x_180, output)
    output = torch.where(is_sing_rot & (xx > yy) & (xx > zz) & (xx >= epsilon), torch.cat([torch.sqrt(xx),
                                                                                           xy4/torch.sqrt(xx),
                                                                                           xz4/torch.sqrt(xx)],
                                                                                          dim=-1) * torch.pi, output)
    output = torch.where(is_sing_rot & (yy > xx) & (yy > zz) & (yy < epsilon), y_180, output)
    output = torch.where(is_sing_rot & (yy > xx) & (yy > zz) & (yy >= epsilon), torch.cat([xy4/torch.sqrt(yy),
                                                                                           torch.sqrt(yy),
                                                                                           yz4/torch.sqrt(yy)],
                                                                                          dim=-1) * torch.pi, output)
    output = torch.where(is_sing_rot & (zz > xx) & (zz > yy) & (zz < epsilon), z_180, output)
    output = torch.where(is_sing_rot & (zz > xx) & (zz > yy) & (zz >= epsilon), torch.cat([xz4/torch.sqrt(zz),
                                                                                           yz4/torch.sqrt(zz),
                                                                                           torch.sqrt(zz)],
                                                                                          dim=-1) * torch.pi, output)

    # No singularity case. Normalize the off-diagonal entries to form the vector, with the angle
    # determined from the rotation matrice's trace.
    trace = torch.clamp(trace, -1 + epsilon, 3 - epsilon)
    s = torch.sqrt(ymz**2 + xmz**2 + xmy**2)
    angle = torch.acos((trace - 1) / 2)
    return torch.where(~is_singular, torch.cat([ymz, xmz, xmy], dim=-1) * angle / s, output)  # noqa


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector (x, y, z, w).

    This implementation is a numerically more stable implementation of the one in kornia.
    https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#rotation_matrix_to_quaternion.
    Instead of adding an eps to avoid negative sqrt roots, the values are clamped to eps.

    Args:
        rotation_matrix: the rotation matrix to convert with shape :math:`(..., 3, 3)`.
        eps: small value to avoid zero division.
    Return:
        the rotation in quaternion with shape :math:`(..., 4)`.
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")
    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    shape = rotation_matrix.shape[:-2]
    if len(shape) == 0:  # input tensor two-dimensional
        rotation_matrix_flat = rotation_matrix[None]
    else:
        rotation_matrix_flat = torch.flatten(rotation_matrix, end_dim=-3)
    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix_flat.shape[:-2], 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)
    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(torch.clamp_min(trace + 1.0, min=eps)) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qx, qy, qz, qw), dim=-1)

    def cond_1():
        sq = torch.sqrt(torch.clamp_min(1.0 + m00 - m11 - m22, min=eps)) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qx, qy, qz, qw), dim=-1)

    def cond_2():
        sq = torch.sqrt(torch.clamp_min(1.0 + m11 - m00 - m22, eps)) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qx, qy, qz, qw), dim=-1)

    def cond_3():
        sq = torch.sqrt(torch.clamp_min(1.0 + m22 - m00 - m11, min=eps)) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qx, qy, qz, qw), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())  # noqa
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)  # noqa
    quaternion_flat = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion_flat.view(*shape, 4)


def rotation_6d_to_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix. Based on Zhou et al., "On the Continuity of
    Rotation Representations in Neural Networks", CVPR 2019

    >>> x_test = torch.rand((4, 8, 6))
    >>> y_test = rotation_6d_to_rotation_matrix(x_test)
    >>> assert y_test.shape == (4, 8, 3, 3)

    Args:
        x: (B,N,6) Batch of 6-D rotation representations
    Returns:
        (B,N,3,3) Batch of corresponding rotation matrices
    """
    x_shape = x.shape
    x = x.view(-1, 3, 2)

    a1 = x[..., 0]
    a2 = x[..., 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)

    rotmat = torch.stack((b1, b2, b3), dim=-1)
    return rotmat.view(*x_shape[:-1], 3, 3)


def rotation_6d_to_quaternion(x: torch.Tensor) -> torch.Tensor:
    """Convert 6d rotation representation to quaternion representation.

    Args:
        x: 6d rotation tensor (..., 6).
    """
    x = rotation_6d_to_rotation_matrix(x)
    x2 = x * 0.9999  # safe gradient (see https://github.com/kornia/kornia/issues/1653)
    return rotation_matrix_to_quaternion(x2)


def rotation_6d_to_axis_angle(x: torch.Tensor) -> torch.Tensor:
    """Convert 6d rotation representation to axis angle (3D) representation.

    Args:
        x: 6d rotation tensor (..., 6)
    """
    x4d = rotation_6d_to_quaternion(x)
    return quaternion_to_angle_axis(x4d)


def angle_axis_to_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape[:-1]
    if len(shape) == 0:
        x_flat = x[None]
    else:
        x_flat = torch.flatten(x, end_dim=-2)
    R_flat = kornia.geometry.angle_axis_to_rotation_matrix(x_flat)
    return R_flat.view(*shape, 3, 3)


def rotation_matrix_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6d representation.
    from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    batch_dim = x.size()[:-2]
    return x[..., :2].clone().reshape(batch_dim + (6,))


def angle_axis_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation in axis-angle representation to 6d representation."""
    shape = x.shape[:-1]
    if len(shape) == 0:
        x_flat = x[None]
    else:
        x_flat = torch.flatten(x, end_dim=-2)
    y = kornia.geometry.angle_axis_to_rotation_matrix(x_flat)
    y6d = rotation_matrix_to_rotation_6d(y)
    return y6d.view(*shape, 6)


def inverse_transformation(x: torch.Tensor) -> torch.Tensor:
    """Inverse transformation tensor i.e. T_AB => T_BA."""
    if not x.shape[-1] == x.shape[-2] == 4:
        raise ValueError(f"Invalid shape of transformation, expected (..., 4, 4), got {x.shape}")

    x_inv = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    Rx_inv = torch.transpose(x[..., :3, :3], -1, -2)   # same as inverse (rotation matrix)
    x_inv[..., :3, :3] = Rx_inv
    x_inv[..., :3, 3] = - (Rx_inv @ x[..., :3, 3, None])[..., 0]  # - R^1 * t
    x_inv[..., 3, 3] = 1
    return x_inv


def inverse_quaternion(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion, defined as

    q⁻¹ = (qw - qx*x - qy*y - qz*z) / (qw**2 + qx**2 + qy**2 + qz**2)

    Implementation similar to pytorch3d's implementation, find can be found at
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html

    Args:
        q: input quaternion (..., 4), in order (x,y,z,w).
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Invalid shape of quaternion, expected (..., 4), got {q.shape}")
    scaling = torch.tensor([-1, -1, -1, 1], device=q.device, dtype=q.dtype)
    return q * scaling


def multiply_angle_axis(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compose two angle axis.

    Implementation based on mathematical derivation from
    https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations

    Args:
        a: first input angle axis (..., 3).
        b: second input angle axis (..., 3).
    Returns:
        Composed rotation in angle axis, a tensor of angle axes (..., 3).
    """
    if a.shape != b.shape:
        raise ValueError(f"Got non-matching inputs a and b, got {a.shape} and {b.shape}")
    if a.shape[-1] != 3:
        raise ValueError(f"Got invalid axis-angle tensor, expected (..., 3), got {a.shape}")

    alpha_2 = torch.linalg.norm(a, dim=-1, keepdim=True) / 2
    beta_2 = torch.linalg.norm(b, dim=-1, keepdim=True) / 2
    m = a / (alpha_2 * 2 + eps)
    n = b / (beta_2 * 2 + eps)

    sa_sb = torch.sin(alpha_2)*torch.sin(beta_2)
    sa_cb = torch.sin(alpha_2)*torch.cos(beta_2)
    ca_cb = torch.cos(alpha_2)*torch.cos(beta_2)
    ca_sb = torch.cos(alpha_2)*torch.sin(beta_2)

    gamma_2 = torch.acos(ca_cb - sa_sb*torch.sum(m*n, dim=-1, keepdim=True))
    o = (sa_cb*m + ca_sb*n + sa_sb*torch.cross(m, n, dim=-1)) / (torch.sin(gamma_2) + eps)
    return o / o.norm(dim=-1, keepdim=True) * 2 * gamma_2


def multiply_quaternion(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions.

    Implementation similar to pytorch3d's implementation, find can be found at
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html

    Args:
        a: first input quaternion (..., 4), in order (x,y,z,w).
        b: second input quaternion (..., 4), in order (x,y,z,w).
    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    ax, ay, az, aw = torch.unbind(a, -1)
    bx, by, bz, bw = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    q_out = torch.stack((ox, oy, oz, ow), -1)
    return torch.where(q_out[..., 3:4] < 0, -q_out, q_out)


def pose_to_transformation_matrix(t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Convert translation and orientation vector to 4x4 transformation matrix.

    Args:
        t: translation vector [..., 3] or [3].
        q: orientation vector in axis-angle or 6d representation.
    Returns:
        transformation matrix [..., 4, 4]
    """
    assert t.shape[:-1] == q.shape[:-1]
    assert t.shape[-1] == 3
    shape = t.shape[:-1]
    if len(q.shape) == 1:
        q = q[None]

    T = torch.zeros((*shape, 4, 4), device=t.device, dtype=t.dtype)
    T[..., 3, 3] = 1.0
    T[..., :3, 3] = t
    if q.shape[-1] == 3:  # axis-angle representation
        T[..., :3, :3] = angle_axis_to_rotation_matrix(q)
    elif q.shape[-1] == 4:  # quaternions
        T[..., :3, :3] = quaternion_to_rotation_matrix(q)
    elif q.shape[-1] == 6:  # double axis representation
        T[..., :3, :3] = rotation_6d_to_rotation_matrix(q)
    else:
        raise NotImplementedError(f"{q.shape[-1]}-dimensional rotation vector not supported")
    return T
