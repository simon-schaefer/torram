import logging
import torch
import torchvision

from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = ['read_image',
           'read_video',
           'write_jpeg',
           'write_png',
           'read_images',
           'read_video_BHWC',
           'read_video_metadata',
           'write_video',
           'write_video_to_images']


def read_images(image_files: List[Union[Path, str]], sort: bool = False) -> torch.Tensor:
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


def read_image(path: Union[Path, str], mode: torchvision.io.ImageReadMode = torchvision.io.ImageReadMode.UNCHANGED
               ) -> torch.Tensor:
    """Reads a JPEG or PNG image into a 3 dimensional RGB or grayscale Tensor.

    Args:
        path: path of the JPEG or PNG image.
        mode: the read mode used for optionally converting the image.
    Returns:
        output: image tensor (3, H, W) as uint8 ranging from 0 to 255.
    """
    if isinstance(path, Path):
        path = path.as_posix()
    return torchvision.io.read_image(path, mode=mode)


def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Reads a video from a file, returning both the video frames and the audio frames

    Args:
        filename: path to the video file
        start_pts: start presentation time of the video
        end_pts: end presentation time
        pts_unit: unit in which start_pts and end_pts values will be interpreted, either 'pts' or 'sec'.
        output_format: format of the output video tensors. Can be either "THWC" (default) or "TCHW".
    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
        aframes (Tensor[K, L]): the audio frames, where `K` is the number of channels and `L` is the number of points
        info (Dict): metadata for the video and audio. Can contain the fields video_fps (float) and audio_fps (int)
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()
    return torchvision.io.read_video(filename, start_pts, end_pts, pts_unit, output_format=output_format)


def read_video_BHWC(data_file: Union[Path, str], start_index: int = 0, end_index: int = None) -> torch.Tensor:
    """Read video data from file as (B, C, H, W) tensor.

    Returns:
        video_data: images tensor (B, C, H, W).
        video_dict: video metadata (such as fps).
    """
    video_data, _, _ = read_video(data_file, start_pts=start_index, end_pts=end_index)
    return torch.permute(video_data, (0, 3, 1, 2))  # (B, H, W, C) -> (B, C, H, W)


def read_video_metadata(data_file: Union[Path, str]) -> Tuple[int, int, int]:
    """Read metadata from video without having to load it fully, using ffmpeg.

    Returns:
        img_height, img_width, num_frames
    """
    import importlib.util
    ffmpeg_loader = importlib.util.find_spec('ffmpeg')
    if ffmpeg_loader is None:
        raise ImportError("read_video_metadata() requires the ffmpeg library, do `pip install ffmpeg`")

    import ffmpeg
    video_metadata = ffmpeg.probe(data_file)["streams"]
    if len(video_metadata) != 1:
        logging.warning(f"Multiple or no streams detected in video file {data_file}, using zeroth")
    img_height = int(video_metadata[0]["height"])
    img_width = int(video_metadata[0]["width"])
    num_frames = int(video_metadata[0]["duration_ts"])
    return img_height, img_width, num_frames


def write_jpeg(image: torch.Tensor, filename: Union[Path, str], quality: int = 75) -> torch.Tensor:
    """Takes an input tensor in CHW layout and saves it in a JPEG file.

    Args:
        image: int8 image tensor of ``c`` channels, where ``c`` must be 1 or 3.
        filename: Path to save the image.
        quality: Quality of the resulting JPEG file, it must be a number between 1 and 100
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()
    return torchvision.io.write_jpeg(image, filename, quality=quality)


def write_png(image: torch.Tensor, filename: Union[Path, str], compression_level: int = 6):
    """Takes an input tensor in CHW layout (or HW in the case of grayscale images) and saves it in a PNG file.

    Args:
        image: int8 image tensor of ``c`` channels, where ``c`` must be 1 or 3.
        filename: Path to save the image.
        compression_level: Compression factor for the resulting file, it must be a number between 0 and 9.
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()
    return torchvision.io.write_png(image, filename, compression_level=compression_level)


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename: path where the video will be saved
        video_array: tensor containing the individual frames, as an uint8 tensor in [T, H, W, C] format
        fps: video frames per second
        video_codec: the name of the video codec, i.e. "libx264", "h264", etc.
        options: dictionary containing options to be passed into the PyAV video stream
        audio_array: audio tensor (C, N), where C is the number of channels and N is the number of samples
        audio_fps: audio sample rate, typically 44100 or 48000
        audio_codec: the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options: dictionary containing options to be passed into the PyAV audio stream
    """
    if isinstance(filename, Path):
        filename = filename.as_posix()
    return torchvision.io.write_video(
        filename=filename,
        video_array=video_array,
        fps=fps,
        video_codec=video_codec,
        options=options,
        audio_array=audio_array,
        audio_fps=audio_fps,
        audio_codec=audio_codec,
        audio_options=audio_options
    )


def write_video_to_images(video: torch.Tensor, directory: Path) -> List[Path]:
    """Write all frames contained in the video as jpg images.

    Args:
        video: video data tensor (N, C, H, W).
        directory: image directory, images are written at directory/image_00001.jpg with 5 zero padding
    """
    image_files = []
    for k, img in enumerate(video):
        img_file_k = directory / f"image_{k:05}.jpg"
        write_jpeg(img, img_file_k)
        image_files.append(img_file_k)
    return image_files
