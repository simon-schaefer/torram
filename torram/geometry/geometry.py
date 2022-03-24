import kornia
import torch
import kornia.geometry.conversions
from kornia.geometry import (
    angle_axis_to_quaternion,
    convert_points_to_homogeneous,
    quaternion_to_angle_axis,
    rotation_matrix_to_quaternion,
    rotation_matrix_to_angle_axis,
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
           'inverse_transformation',
           'pose_to_transformation_matrix',
           'transform_points',
           'convert_points_to_homogeneous'
           ]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    norm_q = q.norm(dim=1, keepdim=True)
    q_normed = q / norm_q
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_order = kornia.geometry.conversions.QuaternionCoeffOrder.XYZW
        rotmat = kornia.geometry.quaternion_to_rotation_matrix(q_normed, order=q_order)
    return rotmat


def rotation_6d_to_rotation_matrix(x):
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
    https://stackoverflow.com/questions/12463487/obtain-rotation-axis-from-rotation-matrix-and-translation-vector-in-opencv

    Args:
        x: 6d rotation tensor (..., 6)
    """
    assert x.shape[-1] == 6
    r = rotation_6d_to_rotation_matrix(x)

    angle = torch.arccos((r[..., 0, 0] + r[..., 1, 1] + r[..., 2, 2] - 1)/2)
    yz = (r[..., 2, 1] - r[..., 1, 2])**2
    xz = (r[..., 0, 2] - r[..., 2, 0])**2
    xy = (r[..., 1, 0] - r[..., 0, 1])**2
    norm = torch.sqrt(xy + xz + yz)

    ax = (r[..., 2, 1] - r[..., 1, 2]) / norm * angle
    ay = (r[..., 0, 2] - r[..., 2, 0]) / norm * angle
    az = (r[..., 1, 0] - r[..., 0, 1]) / norm * angle
    return torch.stack([ax, ay, az], dim=-1)


def angle_axis_to_rotation_matrix(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape[:-1]
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
    x_inv[..., :3, 3] = - torch.matmul(Rx_inv, x[..., :3, 3, None])[..., 0]  # - R^1 * t
    x_inv[..., 3, 3] = 1
    return x_inv


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
