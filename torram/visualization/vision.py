import kornia
import torch
import torram

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from typing import Tuple


__all__ = ['draw_bounding_boxes',
           'draw_segmentation_masks',
           'draw_keypoints',
           'draw_reprojection',
           'draw_keypoints_weighted']


@torch.no_grad()
def draw_keypoints(image: torch.Tensor, points: torch.Tensor, color: Tuple[int, int, int] = (255, 0, 0)
                   ) -> torch.Tensor:
    """Draw keypoints in pixel coordinates in image by coloring the pixel.

    Args:
        image: base image to draw keypoints in (3, H, W).
        points: keypoints in pixel coordinates (N, 2).
        color: keypoint color as RGB tuple (0-255).
    """
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"Keypoints have invalid shape, expected (N, 2), got {points.shape}")
    if image.dtype != torch.uint8:
        raise ValueError(f"Invalid image type, expected uint8, got {image.dtype}")
    if not all(0 <= x <= 255 for x in color):
        raise ValueError(f"Invalid values in color tuple, must be [0, 255], got {color}")

    # Remove the points that are not in the image.
    h, w = image.shape[-2:]
    is_in_image = torram.geometry.is_in_image(points, height=h, width=w)
    points_in = points[is_in_image, :]

    # Color the remaining points with the given color.
    out_image = image.clone()
    for i in range(3):
        out_image[i, points_in[:, 1], points_in[:, 0]] = color[i]
    return out_image


@torch.no_grad()
def draw_reprojection(pc_C: torch.Tensor, image: torch.Tensor, K: torch.Tensor,
                      color: Tuple[int, int, int] = (255, 0, 0)) -> torch.Tensor:
    """Re-Project a point cloud in the camera frame to the image plane and draw the points.

    Args:
        pc_C: point cloud in camera frame (N, 3).
        image: base image to color pixels in.
        K: camera intrinsics for re-projection (3, 3).
        color: color of colored pixels, either as RGB tuple or hex color (uniform color).
    """
    if pc_C.ndim != 2 or pc_C.shape[-1] != 3:
        raise ValueError(f"Point clouds have invalid shape, expected (B, N, 3), got {pc_C.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics have invalid shape, expected (3, 3), got {K.shape}")
    K_point_cloud = K[None, :, :]
    pc_projections = kornia.geometry.project_points(pc_C, camera_matrix=K_point_cloud).long()  # int image coordinates
    return draw_keypoints(image.detach().cpu(), pc_projections, colors=color)


@torch.no_grad()
def draw_keypoints_weighted(image: torch.Tensor, keypoints: torch.Tensor, scores: torch.Tensor,
                            min_color: Tuple[int, int, int], max_color: Tuple[int, int, int], radius: int = 1
                            ) -> torch.Tensor:
    """Draw keypoints in image with different weight. The larger the score the more the color of the keypoints
    will shift to the max color.

    Args:
        image: base image (3, H, W).
        keypoints: points to draw in image (N, 2).
        scores: weights for color evaluation, (0 <= scores <= 1).
        min_color: color at score = 0 (R, G, B).
        max_color: color at score = 1 (R, G, B).
        radius: keypoint radius.
    """
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("Invalid scores, they must be in [0, 1]")
    if len(scores.shape) != 1:
        raise ValueError(f"Invalid scores shape, expected (N, ), got {scores.shape}")
    if len(keypoints.shape) != 2 or len(keypoints) != len(scores):
        raise ValueError(f"Not matching keypoint and scores, got {keypoints.shape} and {scores.shape}")

    for keypoint_k, score_k in zip(keypoints, scores):
        color_k = (int(min_color[0] * (1 - score_k) + max_color[0] * score_k),
                   int(min_color[1] * (1 - score_k) + max_color[1] * score_k),
                   int(min_color[2] * (1 - score_k) + max_color[2] * score_k))
        image = draw_keypoints(image, keypoint_k[None, None], colors=color_k, radius=radius)
    return image
