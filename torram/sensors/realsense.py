import numpy as np
import torch

from functools import partial
from importlib.util import find_spec
from typing import Callable, Optional, Tuple


class RealsenseRGBDriver:
    """Streaming RGB images from Realsense using the official Realsense SDK.

    This implementation has been tested on the Realsense D455. Other devices might not be supported.
    """
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        if find_spec('pyrealsense2') is None:
            raise ImportError("PyRealsense2 module not found, install it using `pip install pyrealsense2` or "
                              "by building and installing librealsense from source")
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._is_streaming = False

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        if not any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors):
            raise Exception("Driver requires device with RGB camera")
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def start_streaming(self):
        self.pipeline.start(self.config)
        self._is_streaming = True

    def stop_streaming(self):
        self.pipeline.stop()
        self._is_streaming = False

    def wait_for_frame_np(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """Request frame from realsense pipeline and retrieve RGB frame, intrinsics and timestamp.
        If the frame does not contain RGB data, return None. Returns numpy arrays.

        Returns:
            color_frame: RGB frame (3, H, W).
            K: camera intrinsics (3, 3).
            timestamp: recording timestamp in seconds.
        """
        if not self._is_streaming:
            self.start_streaming()

        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        if color_frame is None:
            return None
        color_frame = np.asanyarray(color_frame.get_data())
        color_frame = color_frame.transpose((2, 0, 1))  # (H, W, 3) -> (3, H, W)

        intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
        K = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
        timestamp = frame.get_timestamp()

        return color_frame, K, timestamp

    def wait_for_frame(self, device: torch.device, dtype: torch.dtype = torch.float32):
        """Request frame from realsense pipeline and retrieve RGB frame, intrinsics and timestamp.
        If the frame does not contain RGB data, return None. Returns torch.tensors.

        Returns:
            color_frame: RGB frame (3, H, W).
            K: camera intrinsics (3, 3).
            timestamp: recording timestamp in seconds.
        """
        frame = self.wait_for_frame_np()
        if frame is None:
            return None
        color_frame, K, timestamp = frame
        color_frame = torch.tensor(color_frame, device=device, dtype=torch.uint8)
        K = torch.tensor(K, device=device, dtype=dtype)
        return color_frame, K, timestamp

    def _stream(self, processing_func, get_frame_func):
        self.start_streaming()
        try:
            while True:
                frame = get_frame_func()
                if frame is None:
                    continue
                color_frame, K, timestamp = frame
                processing_func(color_frame, K, timestamp)
        finally:
            self.stop_streaming()

    def stream_np(self, processing_func: Callable[[np.ndarray, np.ndarray, float], None]):
        """Stream RGB images, its intrinsics and recording timestamp to processing functions.

        Request RGB images from realsense pipeline. Every time a new frame arrives convert it to a numpy array,
        retrieve its metadata and send it to the processing function. The "wait for new frame" as well as the
        processing operation are blocking (!), so not running asynchronously.

        Args:
            processing_func: function to process every frame.
        """
        self._stream(processing_func, get_frame_func=self.wait_for_frame_np)

    def stream(
        self,
        processing_func: Callable[[torch.Tensor, torch.Tensor, float], None],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Stream RGB images, its intrinsics and recording timestamp to processing functions.

        Request RGB images from realsense pipeline. Every time a new frame arrives convert it to a torch tensor,
        retrieve its metadata and send it to the processing function. The "wait for new frame" as well as the
        processing operation are blocking (!), so not running asynchronously.

        Args:
            processing_func: function to process every frame.
            device: torch device to load frame to.
            dtype: metadata torch dtype.
        """
        wait_for_frame_w_device = partial(self.wait_for_frame, device=device, dtype=dtype)
        self._stream(processing_func, get_frame_func=wait_for_frame_w_device)
