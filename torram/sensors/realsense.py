import numpy as np
import torch

from importlib.util import find_spec
from typing import Callable


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

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        if not any(s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors):
            raise Exception("Driver requires device with RGB camera")
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    def stream_np(self, processing_func: Callable[[np.ndarray, np.ndarray, float], None]):
        """Stream RGB images, its intrinsics and recording timestamp to processing functions.

        Request RGB images from realsense pipeline. Every time a new frame arrives convert it to a numpy array,
        retrieve its metadata and send it to the processing function. The "wait for new frame" as well as the
        processing operation are blocking (!), so not running asynchronously.

        Args:
            processing_func: function to process every frame.
        """
        self.pipeline.start(self.config)
        try:
            while True:
                frame = self.pipeline.wait_for_frames()
                color_frame = frame.get_color_frame()
                if color_frame is None:
                    continue
                color_frame = np.asanyarray(color_frame.get_data())
                color_frame = color_frame.transpose((2, 0, 1))  # (H, W, 3) -> (3, H, W)

                intrinsics = frame.get_profile().as_video_stream_profile().get_intrinsics()
                K = np.array([[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]])
                timestamp = frame.get_timestamp()

                processing_func(color_frame, K, timestamp)
        finally:
            self.pipeline.stop()

    def stream(
        self,
        processing_func: Callable[[torch.Tensor, torch.Tensor, float], None],
        device: torch.device,
        dtype: torch.dtype,
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
        def processing_w_tf(image: np.ndarray, K: np.ndarray, timestamp: float):
            image = torch.tensor(image, device=device, dtype=torch.uint8)
            K = torch.tensor(K, device=device, dtype=dtype)
            processing_func(image, K, timestamp)

        self.stream_np(processing_w_tf)
