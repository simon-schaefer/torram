from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional
from kornia.geometry import depth_to_3d, project_points

__all__ = [
    "depth_to_3d",
    "crop_patches",
    "project_points",
    "is_in_image",
    "box_including_2d",
    "boxes_to_masks",
    "meshes_to_masks",
    "pad",
    "warp",
]


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


def boxes_to_masks(bounding_boxes: torch.Tensor, image_shape: Tuple[int, int]) -> torch.Tensor:
    """Convert bounding boxes to image masks with specified width and height.

    @param bounding_boxes: input bounding boxes (..., 4), int.
    @param image_shape: (width, height) of corresponding image.

    @returns mask with True inside the bounding boxes, False elsewhere (..., height, width).
    """
    assert bounding_boxes.shape[-1] == 4
    assert not (
        torch.is_floating_point(bounding_boxes) or torch.is_complex(bounding_boxes)
    )  # int type
    bbox_flat = torch.flatten(bounding_boxes, end_dim=-2)
    num_bboxes = len(bbox_flat)
    width, height = image_shape

    masks = torch.zeros((num_bboxes, height, width), dtype=torch.bool, device=bounding_boxes.device)
    for i in range(num_bboxes):
        masks[i, bbox_flat[i, 1] : bbox_flat[i, 3], bbox_flat[i, 0] : bbox_flat[i, 2]] = True
    return masks.view(*bounding_boxes.shape[:-1], height, width).contiguous()


def meshes_to_masks(
    vertices: torch.Tensor,
    faces: np.ndarray,
    K: torch.Tensor,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Convert 3D meshes (vertices + faces) to boolean masks using convex hulling.

    @param vertices: 3D mesh vertices in the camera frame (B, N, 3).
    @param faces: 3D mesh faces, same over batch.
    @param K: camera intrinsics, same over batch (3, 3).
    @param image_shape: (image width, image_height).

    @returns masks (B, image height, image width).
    @returns rendered colors (B, 3, image height, image width).
    """
    import scipy

    vertices_2d = project_points(vertices, camera_matrix=K)
    valid_vertices = is_in_image(vertices_2d, width=image_shape[0], height=image_shape[1])
    points = vertices_2d[valid_vertices, :].detach().cpu().numpy()
    deln = scipy.spatial.Delaunay(points)

    idx_2d = np.indices(image_shape, np.int16)
    idx_2d = np.moveaxis(idx_2d, 0, -1)
    deln_mask = deln.find_simplex(idx_2d)
    return ~(deln_mask < 0).T
