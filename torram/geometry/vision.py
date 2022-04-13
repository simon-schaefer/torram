import torch
import torch.nn.functional
from kornia.geometry import depth_to_3d, project_points
from typing import Tuple, Union

__all__ = ['depth_to_3d',
           'project_points',
           'is_in_image',
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
