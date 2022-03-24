import logging
import os
import torch

from torchvision.io import read_image, read_video, write_jpeg, write_png, write_video
from typing import List, Tuple

__all__ = ['read_image',
           'read_video',
           'write_jpeg',
           'write_png',
           'read_images',
           'read_video_BHWC',
           'read_video_metadata',
           'write_video',
           'write_video_to_images']


def read_images(image_files: List[str], sort: bool = False) -> torch.Tensor:
    """Read all image files to a single stacked tensor of shape (N, 3, H, W).

    Assumptions:
    - All images have the same shape, which is the shape of the first image file.
    - When `sort` is True, the image filenames are in some kind of feasibly sortable order.
    """
    if len(image_files) == 0:
        raise ValueError("Got empty list of image files.")
    if sort:
        image_files = sorted(image_files)

    frame_0 = read_image(image_files[0])
    images = torch.zeros((len(image_files), *frame_0.shape), dtype=frame_0.dtype, device=frame_0.device)
    images[0] = frame_0

    for k, img_file in enumerate(image_files[1:]):
        images[k+1] = read_image(img_file)
    return images


def read_video_BHWC(data_file: str, start_index: int = 0, end_index: int = None) -> torch.Tensor:
    """Read video data from file as (B, C, H, W) tensor.

    Returns:
        video_data: images tensor (B, C, H, W).
        video_dict: video metadata (such as fps).
    """
    video_data, _, _ = read_video(data_file, start_pts=start_index, end_pts=end_index)
    return torch.permute(video_data, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)


def read_video_metadata(data_file: str) -> Tuple[int, int, int]:
    """Read metadata from video without having to load it fully, using ffmpeg.

    Returns:
        img_height, img_width, num_frames
    """
    import ffmpeg
    video_metadata = ffmpeg.probe(data_file)["streams"]
    if len(video_metadata) != 1:
        logging.warning(f"Multiple or no streams detected in video file {data_file}, using zeroth")
    img_height = int(video_metadata[0]["height"])
    img_width = int(video_metadata[0]["width"])
    num_frames = int(video_metadata[0]["duration_ts"])
    return img_height, img_width, num_frames


def write_video_to_images(video: torch.Tensor, directory: str) -> List[str]:
    """Write all frames contained in the video as jpg images.

    Args:
        video: video data tensor (N, C, H, W).
        directory: image directory, images are written at directory/image_00001.jpg with 5 zero padding
    """
    image_files = []
    for k, img in enumerate(video):
        img_file_k = os.path.join(directory, f"image_{k:05}.jpg")
        write_jpeg(img, img_file_k)
        image_files.append(img_file_k)
    return image_files
