import enum
import kornia
import torch
import torch.nn.functional
import kornia.geometry.conversions

from kornia.geometry import (
    convert_points_to_homogeneous,
    transform_points
)

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
           'convert_points_to_homogeneous',
           'convert_rotation',
           'interpolate2d',
           'Rotations'
           ]


class Rotations(enum.Enum):
    AXIS_ANGLE = 0
    QUATERNION = 1
    ROTATION6D = 2
    MATRIX = 3


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert a quaternion (x, y, z, w) to a rotation matrix. Adapted from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#quaternion_to_rotation_matrix

    >>> quaternion = torch.tensor((0., 0., 0., 1.))
    >>> quaternion_to_rotation_matrix(quaternion)
    tensor([[-1.,  0.,  0.],
            [ 0., -1.,  0.],
            [ 0.,  0.,  1.]])

    Args:
        q: a tensor containing a quaternion to be converted (..., 4), in XYZW order.
    Return:
        the rotation matrix of shape :math:`(..., 3, 3)`.
    """
    if not isinstance(q, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(q)}")
    if not q.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape (*, 4). Got {q.shape}")

    # normalize the input quaternion
    quaternion_norm: torch.Tensor = torch.nn.functional.normalize(q, p=2.0, dim=-1)

    # unpack the normalized quaternion components
    x, y, z, w = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.ones_like(x)

    return torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(*q.shape[:-1], 3, 3)


def quaternion_to_angle_axis(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector (x, y, z, w) to angle axis of rotation in radians. Adapted from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#quaternion_to_angle_axis

    >>> quaternion = torch.tensor((0., 0., 0., 1.))
    >>> quaternion_to_angle_axis(quaternion)
    tensor([3.1416, 0.0000, 0.0000])

    Args:
        q: tensor with quaternions (..., 4), in XYZW order.
    Return:
        angle axis rotation vector (..., 3).
    """
    if not torch.is_tensor(q):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(q)}")
    if not q.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {q.shape}")

    q1 = q[..., 0]
    q2 = q[..., 1]
    q3 = q[..., 2]
    cos_theta = q[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(q)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def angle_axis_to_quaternion(a: torch.Tensor) -> torch.Tensor:
    """Convert an angle axis to a quaternion (x, y, z, w). Adapted from
    https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#angle_axis_to_quaternion

    >>> x = torch.tensor((0., 1., 0.))
    >>> angle_axis_to_quaternion(x)
    tensor([0.0000, 0.4794, 0.0000, 0.8776])

    Args:
        a: tensor with angle axis in radians (..., 3)
    Return:
        tensor with quaternion (..., 4), in XYZW order.
    """
    if not torch.is_tensor(a):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(a)}")
    if not a.shape[-1] == 3:
        raise ValueError(f"Input must be a tensor of shape Nx3 or 3. Got {a.shape}")

    # unpack input and compute conversion
    a0: torch.Tensor = a[..., 0:1]
    a1: torch.Tensor = a[..., 1:2]
    a2: torch.Tensor = a[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros(size=(*a.shape[:-1], 4), dtype=a.dtype, device=a.device)
    quaternion[..., 0:1] = a0 * k
    quaternion[..., 1:2] = a1 * k
    quaternion[..., 2:3] = a2 * k
    quaternion[..., 3:4] = w
    return quaternion


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
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1, dim=-1)
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


def angle_axis_to_rotation_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.
    Code adapted & simplified from kornia.geometry.angle_axis_to_rotation_matrix

    Args:
        x: axis-angle vector in radians to convert (*, 3).
        eps: precision number.
    Returns:
        rotation matrix representations of x (*, 3, 3).
    """
    if not x.shape[-1] == 3:
        raise ValueError(f"Input size must be a (*, 3) tensor. Got {x.shape}")

    def _compute_rotation_matrix(angle_axis, theta2):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise,
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=-1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
        return rotation_matrix.view(*angle_axis.shape[:-1], 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=-1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=-1)
        return rotation_matrix.view(*angle_axis.shape[:-1], 3, 3)

    # stolen from ceres/rotation.h
    _angle_axis = torch.unsqueeze(x, dim=-2)
    _theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(-1, -2))
    _theta2 = torch.squeeze(_theta2, dim=-2)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(x, _theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(x)

    # create mask to handle both cases
    mask = (_theta2 > eps).view(*x.shape[:-1], 1, 1).to(_theta2.device)
    mask_pos = mask.type_as(_theta2)
    mask_neg = (~mask).type_as(_theta2)

    return mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor


def rotation_matrix_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6d representation.
    from https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    """
    batch_dim = x.size()[:-2]
    return x[..., :2].clone().reshape(batch_dim + (6,))


def angle_axis_to_rotation_6d(x: torch.Tensor) -> torch.Tensor:
    """Convert rotation in axis-angle representation to 6d representation."""
    y = angle_axis_to_rotation_matrix(x)
    y6d = rotation_matrix_to_rotation_6d(y)
    return y6d


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
        eps: small number of numerically safe division.
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


def convert_rotation(x: torch.Tensor, target: Rotations) -> torch.Tensor:
    if target is Rotations.AXIS_ANGLE:
        if x.shape[-1] == 3:
            return x
        elif x.shape[-1] == 4:
            return quaternion_to_angle_axis(x)
        elif x.shape[-1] == 6:
            return rotation_6d_to_axis_angle(x)

    elif target is Rotations.QUATERNION:
        if x.shape[-1] == 3:
            return angle_axis_to_quaternion(x)
        elif x.shape[-1] == 4:
            return x
        elif x.shape[-1] == 6:
            return rotation_6d_to_quaternion(x)

    elif target is Rotations.ROTATION6D:
        if x.shape[-1] == 3:
            return angle_axis_to_rotation_6d(x)
        elif x.shape[-1] == 4:
            raise NotImplementedError
        elif x.shape[-1] == 6:
            return x

    elif target is Rotations.MATRIX:
        if x.shape[-1] == 3:
            return angle_axis_to_rotation_matrix(x)
        elif x.shape[-1] == 4:
            return quaternion_to_rotation_matrix(x)
        elif x.shape[-1] == 6:
            return rotation_6d_to_rotation_matrix(x)

    raise NotImplementedError(f"No conversion from shape {x.shape} to {target} found")


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
    R = convert_rotation(q, target=Rotations.MATRIX)
    T34 = torch.cat([R, t[..., None]], dim=-1)
    T = torch.cat([T34, torch.zeros((*shape, 1, 4), device=t.device, dtype=t.dtype)], dim=-2)
    T[..., 3, 3] = 1
    return T


def interpolate2d(data: torch.Tensor, points: torch.Tensor, mode: str = "bilinear"):
    if len(data.shape) != 3:
        raise ValueError(f"Invalid shape of data, expected (C, H, W), got {data.shape}")
    if len(points.shape) != 2 or points.shape[-1] != 2:
        raise ValueError(f"Invalid shape of points, expected (N, 2), got {points.shape}")

    _, h, w = data.shape

    x0 = torch.floor(points).long()
    u0 = torch.clamp(x0[:, 0], 0, w - 2)
    v0 = torch.clamp(x0[:, 1], 1, h - 2)
    u1 = u0 + 1
    v1 = v0 + 1

    if mode == "bilinear":
        Ia = data[:, v0, u0]
        Ib = data[:, v1, u0]
        Ic = data[:, v0, u1]
        Id = data[:, v1, u1]

        ud = (points[:, 0] - u0) / (u1 - u0)
        vd = (points[:, 1] - v0) / (v1 - v0)
        zv0 = Ia * (1 - ud) + Ic * ud
        zv1 = Ib * (1 - ud) + Id * ud
        return zv0 * (1 - vd) + zv1 * vd

    else:
        raise NotImplementedError(f"Interpolation mode {mode} not supported")
