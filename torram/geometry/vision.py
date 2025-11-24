from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from kornia.geometry import transform_points
from kornia.geometry.camera.perspective import project_points, unproject_points

__all__ = ["crop_patches", "is_in_image", "box_including_2d", "pad", "warp", "warp_depth_image"]


def is_in_image(
    pixel: torch.Tensor, width: Union[int, torch.Tensor], height: Union[int, torch.Tensor]
) -> torch.Tensor:
    """Check which pixels are in the image.

    @param pixel: pixels in image [..., 2].
    @param width: image width.
    @param height: image height.

    @returns both pixel coordinates are in the image [...].
    """
    is_in_image_u = torch.logical_and(pixel[..., 0] >= 0, pixel[..., 0] < width)
    is_in_image_v = torch.logical_and(pixel[..., 1] >= 0, pixel[..., 1] < height)
    return torch.logical_and(is_in_image_u, is_in_image_v)


def warp(points: torch.Tensor, warping: torch.Tensor) -> torch.Tensor:
    """Warp 2D image coordinates with warping tensor.

    @param points: image coordinates (..., M, 2).
    @param warping: warping matrix (..., 3, 3).

    @returns warping image coordinates (..., M, 2).
    """
    assert points.ndim >= 2 and points.shape[-1] == 2
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=-1).to(warping.dtype)
    points_warped = torch.einsum("...il,...ml->...mi", warping, points_h)
    points_warped = points_warped[..., :2] / points_warped[..., -1, None]
    return points_warped.to(points.dtype)


def warp_depth_image(
    depth_img: torch.Tensor,
    K_src: torch.Tensor,
    K_tgt: torch.Tensor,
    img_size_tgt: Tuple[int, int],
    T_src_tgt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Warp depth image from source to target view.

    @param depth_img: source depth image (B, H, W).
    @param K_src: source camera intrinsics (B, 3, 3).
    @param K_tgt: target camera intrinsics (B, 3, 3).
    @param img_size_tgt: target image size (Wt, Ht).
    @param T_tgt_src: transformation from source to target camera coordinates (B, 4, 4).

    @returns warped depth image in target view (B, Ht, Wt).
    """
    B, H, W = depth_img.shape
    W_new, H_new = img_size_tgt
    depth = depth_img[:, None, :, :]

    # Build pixel grid in *target* resolution
    y, x = torch.meshgrid(
        torch.arange(H_new, device=depth.device),
        torch.arange(W_new, device=depth.device),
        indexing="ij",
    )
    pix_tgt = torch.stack([x, y], dim=-1).float()
    pix_tgt = pix_tgt.unsqueeze(0).repeat(B, 1, 1, 1)

    # Convert target pixels → target camera rays
    fx_t = K_tgt[:, 0, 0].view(B, 1, 1)
    fy_t = K_tgt[:, 1, 1].view(B, 1, 1)
    cx_t = K_tgt[:, 0, 2].view(B, 1, 1)
    cy_t = K_tgt[:, 1, 2].view(B, 1, 1)

    x_norm = (pix_tgt[..., 0] - cx_t) / fx_t
    y_norm = (pix_tgt[..., 1] - cy_t) / fy_t
    rays_tgt = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1)

    # Transform rays from target camera → source camera using Kornia
    if T_src_tgt is not None:
        rays_tgt_flat = rays_tgt.view(B, -1, 3)
        rays_src_flat = transform_points(T_src_tgt, rays_tgt_flat)
        rays_src = rays_src_flat.view(B, H_new, W_new, 3)
    else:
        rays_src = rays_tgt

    # Project rays_src into the *source* image plane
    pts_proj_src = project_points(rays_src, K_src)
    u_src = pts_proj_src[..., 0]
    v_src = pts_proj_src[..., 1]

    # Sample depth values from source depth image. Use grid_sample which requires
    # normalized coordinates in [-1, 1]. Use nearest neighbor interpolation to avoid
    # artifacts (no interpolation between depth values & holes).
    u_norm = (u_src / (W - 1)) * 2 - 1
    v_norm = (v_src / (H - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=-1)  # (B,H_new,W_new,2)

    return F.grid_sample(
        depth,
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )


def depth_image_from_points(
    points: Float[np.ndarray, "N 3"],
    camera_matrix: Float[np.ndarray, "3 3"],
    img_size: Tuple[int, int],  # (Wt, Ht)
    T_tgt_src: Optional[Float[np.ndarray, "4 4"]] = None,
) -> Float[np.ndarray, "Ht Wt"]:
    """
    Project a 3D point cloud into a target camera depth image.

    @param points: 3D points in *source* camera coordinates.
    @param camera_matrix: target camera intrinsics.
    @param img_size: (W_tgt, H_tgt)
    @param T_tgt_src: transforms points from source → target camera.
    @returns depth map in target view.
    """
    Wt, Ht = img_size
    N = points.shape[0]

    # Transform points to target camera coordinates.
    if T_tgt_src is not None:
        pts_h = np.concatenate([points, np.ones((N, 1))], axis=-1)  # (N,4)
        pts_3d_tgt = (T_tgt_src @ pts_h.T).T[:, :3]
    else:
        pts_3d_tgt = points

    Z = pts_3d_tgt[:, 2]

    # Project points into image.
    x = pts_3d_tgt[:, 0]
    y = pts_3d_tgt[:, 1]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = fx * x / Z + cx
    v = fy * y / Z + cy

    u = np.round(u).astype(np.int64)
    v = np.round(v).astype(np.int64)

    # Valid pixels (in front of camera and inside image).
    valid = (Z > 0) & (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)

    # Output depth map using z-buffering (rasterization).
    depth_out = np.full((Ht, Wt), np.inf, dtype=float)
    uu = u[valid]
    vv = v[valid]
    zz = Z[valid]
    for px, py, pz in zip(uu, vv, zz):
        if pz < depth_out[py, px]:
            depth_out[py, px] = pz

    # Replace empty pixels with 0.
    depth_out[np.isinf(depth_out)] = 0.0
    return depth_out


def pad(
    images: torch.Tensor, K: torch.Tensor, output_shape: Tuple[int, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad images with zeros to output shape and adapt the intrinsics accordingly. Changing the image center
    does not affect the projection, therefore we just have to translate the image center accordingly.

    @param images: image tensor to pad (B, C, H, W).
    @param K: according intrinsics (B, 3, 3).
    @param output_shape: padded images shape (h, w).
    """
    assert K.shape[-1] == K.shape[-2] == 3
    assert len(output_shape) == 2
    assert all(x > 0 for x in output_shape)

    h, w = images.shape[-2:]
    dh = output_shape[0] - h
    dw = output_shape[1] - w
    images_padded = torch.nn.functional.pad(images, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))

    K_padded = K.clone()
    K_padded[..., 0, 2] += dw // 2
    K_padded[..., 1, 2] += dh // 2
    return images_padded, K_padded


@torch.jit.script
def crop_patches(
    images: torch.Tensor, points: torch.Tensor, width: int, height: int
) -> torch.Tensor:
    """Crop patches from center coordinate with size (2*height, 2*width). If a part of the patch is
    outside the image, zero padding is used.

    @param images: base images (B, 3, H, W).
    @param points: center points of cropping in image coordinates (B, N, 2).
    @param width: number of pixels between center and left/right side of the cropping.
    @param height: number of pixels between center and top/bottom side of the cropping.

    @returns patches: (B, N, 2*height, 2*width).
    """
    assert images.ndim == 4 and images.shape[-3] == 3  # (B, 3, H, W)
    assert points.ndim == 3 and points.shape[-1] == 2  # (B, N, 2)
    assert images.shape[:-3] == points.shape[:-2]
    assert width > 0 and height > 0

    batch_size, _, img_height, img_width = images.shape
    _, num_patches, _ = points.shape
    patches = torch.zeros(
        (batch_size, num_patches, 3, 2 * height, 2 * width),
        dtype=images.dtype,
        device=images.device,
    )
    for k in range(batch_size):
        for n in range(num_patches):
            x_min = torch.clamp(points[k, n, 0] - width, 0, img_width)
            x_max = torch.clamp(points[k, n, 0] + width, 0, img_width)
            y_min = torch.clamp(points[k, n, 1] - height, 0, img_height)
            y_max = torch.clamp(points[k, n, 1] + height, 0, img_height)

            x1 = x_min - (points[k, n, 0] - width)
            x2 = x_max - (points[k, n, 0] - width)
            y1 = y_min - (points[k, n, 1] - height)
            y2 = y_max - (points[k, n, 1] - height)

            patches[k, n, :, y1:y2, x1:x2] = images[k, :, y_min:y_max, x_min:x_max]
    return patches


def box_including_2d(
    points_2d: torch.Tensor,
    x_min: Optional[int] = None,
    y_min: Optional[int] = None,
    x_max: Optional[int] = None,
    y_max: Optional[int] = None,
    offset: int = 0,
) -> torch.Tensor:
    """Compute the smallest rectangle that is in the given bounds and includes all the 2D points.

    @param points_2d: image points to contain [..., 2].
    @param x_min: minimal x coordinate.
    @param y_min: minimal y coordinate.
    @param x_max: maximal x coordinate.
    @param y_max: maximal y coordinate.
    @param offset: offset from the smallest possible box (in both directions).

    @returns boxes [..., 4], with [x_min, y_min, x_max, y_max]
    """
    u_min = torch.min(points_2d[..., 0], dim=-1).values - offset
    u_max = torch.max(points_2d[..., 0], dim=-1).values + offset
    v_min = torch.min(points_2d[..., 1], dim=-1).values - offset
    v_max = torch.max(points_2d[..., 1], dim=-1).values + offset

    if x_min is not None or x_max is not None:
        u_min = torch.clamp(u_min, x_min, x_max)
        u_max = torch.clamp(u_max, x_min, x_max)
    if y_min is not None or y_max is not None:
        v_min = torch.clamp(v_min, y_min, y_max)
        v_max = torch.clamp(v_max, y_min, y_max)
    return torch.stack([u_min, v_min, u_max, v_max], dim=-1)


def unproject(
    points_2d: Int[torch.Tensor, "B N 2"],
    depth_img: Float[torch.Tensor, "B H W"],
    K: Float[torch.Tensor, "B 3 3"],
    T_W_C: Optional[Float[torch.Tensor, "B 4 4"]] = None,
) -> Tuple[Float[torch.Tensor, "B N 3"], Bool[torch.Tensor, "B N"]]:
    """Unproject 2D points to 3D points in camera coordinates.

    @param points_2d: 2D keypoints.
    @param depth_img: depth frame.
    @param K: camera intrinsics.
    @param T_W_C: optional transformation from world to camera coordinates.
    @returns 3D points in camera coordinates and mask indicating valid points.
    """
    B, N, _ = points_2d.shape
    _, h, w = depth_img.shape

    mask = is_in_image(points_2d, w, h)
    points3d = torch.zeros((B, N, 3), dtype=torch.float32, device=points_2d.device)
    if torch.sum(mask) == 0:
        return points3d, mask

    points_2d_flat = points_2d.view(B * N, 2).long()
    mask_flat = mask.view(B * N)
    batch_idx = torch.arange(B, device=points_2d.device).repeat_interleave(N)
    batch_idx = batch_idx[mask_flat]

    depth = depth_img[batch_idx, points_2d_flat[mask_flat, 1], points_2d_flat[mask_flat, 0]]
    points3d[mask] = unproject_points(
        point_2d=points_2d_flat[mask_flat].unsqueeze(0),
        depth=depth.unsqueeze(1),
        camera_matrix=K[batch_idx].unsqueeze(0),
    )

    # Transform from camera to world coordinates, if given.
    if T_W_C is not None:
        points3d[mask] = transform_points(T_W_C[batch_idx], points3d[mask][:, None])[:, 0]

    # Correct the mask for invalid depth values.
    mask_depth = depth > 0
    mask[mask.clone()] = mask_depth
    points3d[~mask] = 0.0

    return points3d, mask


def pluecker_embeddings(
    points2d: Float[torch.Tensor, "B T N 2"],
    K: Float[torch.Tensor, "B 3 3"],
    T_W_C: Float[torch.Tensor, "B T 4 4"],
) -> Float[torch.Tensor, "B T N 6"]:
    """
    Convert 2D keypoints to world-space Plücker line embeddings.

    @param points2d: 2D keypoints in image coordinates.
    @param K: Camera intrinsics.
    @param T_W_C: Transformation from world to camera coordinates.
    @return World-space Plücker line coordinates [d | m].
    """
    device = points2d.device
    B, T, N, _ = points2d.shape
    R_W_C = T_W_C[:, :, :3, :3]
    c_world = T_W_C[:, :, :3, 3]  # camera center in world coordinates

    # Backproject pixel to 3D direction in camera frame.
    homog = torch.cat([points2d, torch.ones((B, T, N, 1), device=device)], dim=-1)
    K_inv = torch.inverse(K)
    d_cam = torch.einsum("bij,btnj->btni", K_inv, homog)
    d_cam = d_cam / torch.norm(d_cam, dim=-1, keepdim=True)

    # Direction and moment in world coordinates. Transform direction and compute moment.
    d_world = torch.einsum("btij,btnj->btni", R_W_C, d_cam)
    m_world = torch.cross(c_world.unsqueeze(-2).expand_as(d_world), d_world, dim=-1)

    # Form Plücker coordinates. Normalization for numerical stability.
    plucker = torch.cat([d_world, m_world], dim=-1)
    plucker = plucker / torch.norm(plucker[:, :, :, :3], dim=-1, keepdim=True)
    return plucker
