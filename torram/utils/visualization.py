from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
import numpy.typing as npt
from jaxtyping import Float, Int, UInt8
from PIL import Image, ImageDraw


def draw_keypoints(
    images: Int[npt.NDArray[np.uint8], "H W 3"],
    keypoints: Float[npt.NDArray[np.float32], "N 2"],
    edges: Optional[List[Tuple[int, int]]] = None,
    keypoint_scores: Optional[Float[npt.NDArray[np.float32], "N"]] = None,
    keypoint_colors: Union[Tuple[int, int, int], Int[npt.NDArray[np.uint8], "N 3"]] = (255, 0, 0),
    edge_colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (0, 0, 0),
    mask: Optional[np.ndarray] = None,
    keypoint_radius: int = 3,
    edge_thickness: int = 2,
    colormap_name: str = "viridis",
) -> Int[npt.NDArray[np.uint8], "H W 3"]:
    N = keypoints.shape[0]
    E = len(edges) if edges is not None else 0
    if mask is None:
        mask = np.ones((N,), dtype=bool)

    # Prepare the image and keypoints / keypoint colors.
    img_pil = Image.fromarray(images)
    draw = ImageDraw.Draw(img_pil)

    if keypoint_scores is not None:
        scores = np.clip(keypoint_scores, 0, 1)
        cmap = matplotlib.colormaps.get_cmap(colormap_name)
        colors_float = cmap(scores)[:, :3]
        kp_colors = (colors_float * 255).astype(np.uint8)
    else:
        kp_colors = (
            keypoint_colors
            if not isinstance(keypoint_colors, tuple)
            else np.array(keypoint_colors, dtype=np.uint8)[None].repeat(N, axis=0)
        )
    assert kp_colors.shape == (N, 3)

    # Draw edges only if both keypoints are visible.
    if edges is not None:
        ec = edge_colors if isinstance(edge_colors, list) else [edge_colors] * E
        assert len(ec) == len(edges)

        for e_idx, (start, end) in enumerate(edges):
            if not (mask[start] and mask[end]):
                continue
            pt1 = tuple(keypoints[start].astype(int))
            pt2 = tuple(keypoints[end].astype(int))
            draw.line([pt1, pt2], fill=ec[e_idx], width=edge_thickness)

    # Draw visible keypoints as circles with the specified radius.
    for n in range(N):
        if not mask[n]:
            continue
        pt = tuple(keypoints[n].astype(int))
        color = tuple(kp_colors[0]) if kp_colors.shape[0] == 1 else tuple(kp_colors[n])
        x, y = pt
        r = keypoint_radius
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=color)

    return np.array(img_pil)


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
