import torch
import torch.nn.functional
from kornia.geometry import depth_to_3d, project_points
from typing import Optional, Tuple, Union

__all__ = ['depth_to_3d',
           'crop_patches',
           'project_points',
           'is_in_image',
           'box_including_2d',
           'pad']


def is_in_image(pixel: torch.Tensor, width: Union[int, torch.Tensor], height: Union[int, torch.Tensor]) -> torch.Tensor:
    """Check which pixels are in the image.

    Args:
        pixel: pixels in image [..., 2].
        width: image width.
        height: image height.
    Returns:
        both pixel coordinates are in the image [...].
    """
    is_in_image_u = torch.logical_and(pixel[..., 0] >= 0, pixel[..., 0] < width)
    is_in_image_v = torch.logical_and(pixel[..., 1] >= 0, pixel[..., 1] < height)
    return torch.logical_and(is_in_image_u, is_in_image_v)


def pad(images: torch.Tensor, K: torch.Tensor, output_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad images with zeros to output shape and adapt the intrinsics accordingly. Changing the image center
    does not affect the projection, therefore we just have to translate the image center accordingly.

    Args:
        images: image tensor to pad (B, C, H, W).
        K: according intrinsics (B, 3, 3).
        output_shape: padded images shape (h, w).
    """
    if not K.shape[-1] == K.shape[-2] == 3:
        raise ValueError(f"Invalid intrinsics shape, expected (B, 3, 3), got {K.shape}")
    if len(output_shape) != 2 or any(x <= 0 for x in output_shape):
        raise ValueError(f"Invalid output image shape, expected (h, w), got {output_shape}")

    h, w = images.shape[-2:]
    dh = output_shape[0] - h
    dw = output_shape[1] - w
    images_padded = torch.nn.functional.pad(images, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))

    K_padded = K.clone()
    K_padded[..., 0, 2] += dw // 2
    K_padded[..., 1, 2] += dh // 2
    return images_padded, K_padded


@torch.jit.script
def crop_patches(images: torch.Tensor, points: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Crop patches from center coordinate with size (2*height, 2*width). If a part of the patch is
    outside the image, zero padding is used.

    Args:
        images: base images (B, 3, H, W).
        points: center points of cropping in image coordinates (B, N, 2).
        width: number of pixels between center and left/right side of the cropping.
        height: number of pixels between center and top/bottom side of the cropping.
    Returns:
        patches: (B, N, 2*height, 2*width).
    """
    if images.shape[:-3] != points.shape[:-2]:
        raise ValueError(f"Got non-matching images and points tensors, got {images.shape} and {points.shape}")
    if len(images.shape) != 4 or images.shape[-3] != 3:
        raise ValueError(f"Got invalid images, expected (B, 3, H, W), got {images.shape}")
    if len(points.shape) != 3 or points.shape[-1] != 2:
        raise ValueError(f"Got invalid pixel coordinates, expected (B, N, 2), got {points.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Got invalid width/height, expected width/height > 0")

    batch_size, _, img_height, img_width = images.shape
    _, num_patches, _ = points.shape
    patches = torch.zeros((batch_size, num_patches, 3, 2 * height, 2 * width), dtype=images.dtype, device=images.device)
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


def box_including_2d(points_2d: torch.Tensor, x_min: Optional[int] = None, y_min: Optional[int] = None,
                     x_max: Optional[int] = None, y_max: Optional[int] = None, offset: int = 0) -> torch.Tensor:
    """Compute the smallest rectangle that is in the given bounds and includes all the 2D points.

    Args:
        points_2d: image points to contain [..., 2].
        x_min: minimal x coordinate.
        y_min: minimal y coordinate.
        x_max: maximal x coordinate.
        y_max: maximal y coordinate.
        offset: offset from the smallest possible box (in both directions).
    Returns:
        boxes [..., 4].
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
