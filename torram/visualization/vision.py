import matplotlib
import kornia
import torch
import torchvision

import matplotlib.cm as cm
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from typing import Optional, List, Tuple, Union


__all__ = ['draw_bounding_boxes',
           'draw_segmentation_masks',
           'draw_keypoints',
           'draw_reprojection',
           'draw_keypoints_weighted']


@torch.no_grad()
def draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    connectivity: Optional[List[Tuple[int, int]]] = None,
    colors: Optional[Union[str, Tuple[int, int, int]]] = "red",
    radius: int = 2,
    width: int = 3,
) -> torch.Tensor:
    """
    Draws Keypoints on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        keypoints (Tensor): Tensor of shape (num_instances, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        connectivity (List[Tuple[int, int]]]): A List of tuple where,
            each tuple contains pair of keypoints to be connected.
        colors (str, Tuple): The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
        radius (int): Integer denoting radius of keypoint.
        width (int): Integer denoting width of line connecting keypoints.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with keypoints drawn.
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
    return torchvision.utils.draw_keypoints(image, keypoints, connectivity, colors, radius=radius, width=width)


@torch.no_grad()
def draw_reprojection(image: torch.Tensor, pc_C: torch.Tensor, K: torch.Tensor,
                      colors: Union[str, Tuple[int, int, int]] = (255, 0, 0), radius: int = 2) -> torch.Tensor:
    """Re-Project a point cloud in the camera frame to the image plane and draw the points.

    Args:
        pc_C: point cloud in camera frame (N, 3).
        image: base image to color pixels in.
        K: camera intrinsics for re-projection (3, 3).
        colors: color of colored pixels, either as RGB tuple or hex color (uniform color).
        radius: radius of drawn keypoints.
    """
    if pc_C.ndim != 3 or pc_C.shape[-1] != 3:
        raise ValueError(f"Point clouds have invalid shape, expected (B, N, 3), got {pc_C.shape}")
    if K.shape != (3, 3):
        raise ValueError(f"Intrinsics have invalid shape, expected (3, 3), got {K.shape}")
    K_point_cloud = K[None, :, :]
    pc_projections = kornia.geometry.project_points(pc_C, camera_matrix=K_point_cloud).long()  # int image coordinates
    return draw_keypoints(image.detach().cpu(), pc_projections, colors=colors, radius=radius)


@torch.no_grad()
def draw_keypoints_weighted(image: torch.Tensor, keypoints: torch.Tensor, scores: torch.Tensor, radius: int = 1,
                            colormap: str = "rainbow") -> torch.Tensor:
    """Draw keypoints in image colored by their scoring (0 <= score <= 1).

    By default, this function uses the matplotlib colormap 'rainbow', ranging from blue for low values to
    red for high values. The scores are not clamped, instead a value error is thrown if they are not in [0, 1].
    Colormap documentation: https://matplotlib.org/3.5.0/tutorials/colors/colormaps.html

    Args:
        image: base image (3, H, W).
        keypoints: points to draw in image (N, 2).
        scores: weights for color evaluation, (0 <= scores <= 1).
        radius: keypoint radius.
        colormap: name of matplotlib colormap.
    """
    if torch.any(scores < 0) or torch.any(scores > 1):
        raise ValueError("Invalid scores, they must be in [0, 1]")
    if len(scores.shape) != 1:
        raise ValueError(f"Invalid scores shape, expected (N, ), got {scores.shape}")
    if len(keypoints.shape) != 2 or len(keypoints) != len(scores):
        raise ValueError(f"Not matching keypoint and scores, got {keypoints.shape} and {scores.shape}")

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(colormap))
    for keypoint_k, score_k in zip(keypoints, scores):
        color_k = color_map.to_rgba(float(score_k), bytes=True)[:3]  # RGBA -> RGB
        image = draw_keypoints(image, keypoint_k[None, None], colors=color_k, radius=radius)
    return image
