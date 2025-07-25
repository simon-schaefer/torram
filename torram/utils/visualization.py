from typing import List, Optional, Tuple, Union

import kornia
import matplotlib
import matplotlib.cm as cm
import numpy as np
import torch
import torchvision
from jaxtyping import UInt8
from PIL import Image, ImageDraw


@torch.no_grad()
def __draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:
    if len(boxes.shape) == 1:
        boxes = boxes[None]
    return torchvision.utils.draw_bounding_boxes(
        image, boxes, labels, colors, fill, width, font, font_size
    )


@torch.no_grad()
def draw_bounding_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
):
    """
    Draws bounding boxes on a given batch of images or a single image.

    Args:
        images: Tensor of shape ([M,] C, H, W) and dtype uint8.
        boxes: Tensor of size ([M,] N, 4) or (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format.
            Note that the boxes are absolute coordinates with respect to the image. In other words:
            `0 <= xmin < xmax < W` and `0 <= ymin < ymax < H`.
        labels: List containing the labels of bounding boxes, shared over batch.
        colors: List containing the colors of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill: If `True` fills the bounding box with specified color.
        width: Width of bounding box.
        font: A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size: The requested font size in points.

    Returns:
        images: Batch of image tensors or single image tensor of dtype uint8 with bounding boxes plotted ([M,] C, H, W).
    """
    if len(images.shape) == 3:
        assert boxes.ndim in [1, 2]  # (num_boxes, 4)
        return __draw_bounding_boxes(images, boxes, labels, colors, fill, width, font, font_size)
    elif len(images.shape) == 4:
        assert boxes.ndim in [2, 3]  # (N, num_boxes, 4)
        assert images.shape[0] == boxes.shape[0]
        output_images = torch.zeros_like(images)
        for k, (image, bboxes) in enumerate(zip(images, boxes)):
            output_images[k] = __draw_bounding_boxes(
                image, bboxes, labels, colors, fill, width, font, font_size
            )
        return output_images
    else:
        raise ValueError(f"Got neither batch nor single image, got {images.shape}")


@torch.no_grad()
def __draw_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    colors: Union[str, Tuple[int, int, int], np.ndarray, List[Tuple[int, int, int]]] = "red",
    radius: int = 2,
) -> torch.Tensor:
    """Internal drawing function for keypoints in an image. Implementation adapted from:
    https://pytorch.org/vision/stable/_modules/torchvision/utils.html#draw_keypoints

    Args:
        image: single image tensor (3, H, W), as torch.uint8.
        keypoints: keypoint 2D positions (N, K, 2).
        colors: keypoint color as string, RGB tuple, list or numpy array (one RGB tuple for each keypoint).
        radius: keypoint radius.
    """
    if keypoints.ndim == 2:
        keypoints = keypoints[None]
    assert isinstance(image, torch.Tensor) and image.dtype == torch.uint8
    assert image.ndim == 3
    assert image.shape[0] == 3  # RGB image
    assert keypoints.ndim == 3  # (num_instances, K, 2)

    if isinstance(colors, np.ndarray):
        assert colors.ndim == 3
        assert colors.shape[-1] == 3  # (N, K, 3)
        assert colors.shape[:-2] == keypoints.shape[:-2]
    elif isinstance(colors, tuple):
        assert len(colors) == 3  # (R, G, B)
    elif isinstance(colors, list):
        assert all([len(colors) == 3 for color in colors])  # [(R, G, B), ...]
        assert len(colors) == len(keypoints)

    img_to_draw = Image.fromarray(image.permute(1, 2, 0).cpu().numpy())
    draw = ImageDraw.Draw(img_to_draw)
    img_kpts = keypoints.to(torch.int64).tolist()

    for kpt_id, kpt_inst in enumerate(img_kpts):
        for inst_id, kpt in enumerate(kpt_inst):
            x1 = kpt[0] - radius
            x2 = kpt[0] + radius
            y1 = kpt[1] - radius
            y2 = kpt[1] + radius
            if isinstance(colors, np.ndarray):
                color = tuple(colors[kpt_id, inst_id].astype(int))
            elif isinstance(colors, list):
                color = colors[kpt_id]
            else:
                color = colors
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None, width=0)
    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


@torch.no_grad()
def draw_keypoints(
    images: torch.Tensor,
    keypoints: torch.Tensor,
    colors: Union[str, Tuple[int, int, int], np.ndarray, List[Tuple[int, int, int]]] = "red",
    radius: int = 2,
) -> torch.Tensor:
    """
    Draws key-points on a given batch of images or a single image.

    Args:
        images: Tensor of shape ([M,] 3, H, W) and dtype uint8.
        keypoints: Tensor of shape ([M,] N, K, 2) the K keypoints location for each of the N instances,
            in the format [x, y].
        colors: The color can be represented as PIL strings ("red" or "#FF00FF"), as RGB tuples (240, 10, 157),
            as list of RGB tuples (N x (R, G, B)) or numpy.array containing a RGB tuple for each keypoint([M,] N, K, 3).
        radius: Integer denoting radius of keypoint.

    Returns:
        images: Batch of image tensors or single image tensor of dtype uint8 with keypoints drawn. ([M,] C, H, W).
    """
    if len(images.shape) == 3:
        assert keypoints.ndim in [2, 3]  # (N, K, 2)
        return __draw_keypoints(images, keypoints, colors=colors, radius=radius)

    elif len(images.shape) == 4:
        assert keypoints.ndim in [3, 4]  # (M, N, K, 2)
        assert images.shape[0] == keypoints.shape[0]
        assert isinstance(colors, np.ndarray) and colors.shape[0] == images.shape[0]
        if len(keypoints.shape) == 3:
            keypoints = keypoints[:, None]
        if isinstance(colors, np.ndarray) and len(colors.shape) == 3:
            colors = colors[:, None]

        output_images = torch.zeros_like(images)
        if not isinstance(colors, np.ndarray):
            colors = [colors] * len(output_images)
        for k, (image, keypoints_k, colors_k) in enumerate(zip(images, keypoints, colors)):
            output_images[k] = __draw_keypoints(image, keypoints_k, colors=colors_k, radius=radius)
        return output_images

    else:
        raise ValueError(f"Got neither batch nor single image, got {images.shape}")


@torch.no_grad()
def draw_reprojection(
    image: torch.Tensor,
    pc_C: torch.Tensor,
    K: torch.Tensor,
    colors: Union[str, Tuple[int, int, int], np.ndarray, List[Tuple[int, int, int]]] = "red",
    radius: int = 2,
) -> torch.Tensor:
    """Re-Project a point cloud in the camera frame to the image plane and draw the points.

    Args:
        pc_C: point cloud in camera frame (N, 3).
        image: base image to color pixels in.
        K: camera intrinsics for re-projection (3, 3).
        colors: The color can be represented as PIL strings ("red" or "#FF00FF"), as RGB tuples (240, 10, 157),
            as list of RGB tuples (N x (R, G, B)) or numpy.array containing a RGB tuple for each keypoint(N, K, 3).
        radius: radius of drawn keypoints.
    """
    assert pc_C.ndim == 3 and pc_C.shape[-1] == 3  # (B, N, 3)
    assert K.shape == (3, 3)
    K_point_cloud = K[None, :, :]
    pc_projections = kornia.geometry.project_points(
        pc_C, camera_matrix=K_point_cloud
    ).long()  # int image coordinates
    return draw_keypoints(image.detach().cpu(), pc_projections, colors=colors, radius=radius)


@torch.no_grad()
def draw_keypoints_weighted(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    scores: torch.Tensor,
    radius: int = 1,
    colormap: str = "rainbow",
) -> torch.Tensor:
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
    assert torch.all(scores >= 0) and torch.all(scores <= 1)  # [0, 1]
    assert scores.ndim == 1  # (N, )
    assert keypoints.ndim == 2 and keypoints.shape == (scores.shape[0], 2)
    assert image.ndim == 3 and image.shape[0] == 3  # (3, H, W)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    color_map = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap(colormap))
    for keypoint_k, score_k in zip(keypoints, scores):
        color_k = color_map.to_rgba(float(score_k), bytes=True)[:3]  # RGBA -> RGB
        image = draw_keypoints(image, keypoint_k[None, None], colors=color_k, radius=radius)
    return image


def make_video_grid(
    videos: UInt8[np.ndarray, "B N C H W"], grid_size: int
) -> UInt8[np.ndarray, "N C Hout Wout"]:
    """Creates a grid of videos.

    @param videos: Videos as a numpy array of shape (B, N, C, H, W).
    @param grid_size: Width of the grid to arrange videos.
    @return: A single video frame with all videos arranged in a grid.
    """
    B, N, C, H, W = videos.shape
    grid_width = min(grid_size, B)
    grid_height = (B + grid_size - 1) // grid_size

    grid_video = np.zeros((N, C, H * grid_height, W * grid_width), dtype=videos.dtype)
    for i in range(B):
        row = i // grid_width
        col = i % grid_width
        start_h = row * H
        start_w = col * W
        grid_video[:, :, start_h : start_h + H, start_w : start_w + W] = videos[i]

    return grid_video
