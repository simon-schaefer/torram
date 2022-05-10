import os
import torch
import torch.utils.tensorboard
from typing import Any, Dict


class TensorboardY(torch.utils.tensorboard.SummaryWriter):

    def __init__(self, log_dir: str, **kwargs):
        os.makedirs(log_dir)
        super(TensorboardY, self).__init__(log_dir=log_dir, **kwargs)

    def add_images_w_iter(self, tag: str, img: torch.Tensor, global_step: int, **kwargs):
        if img.ndim != 4:
            raise ValueError(f"Invalid input image tensor, expected shape (B, C, H, W), got {img.shape}")
        for k, img_k in enumerate(img):
            self.add_image(f"{tag}/{k+1}", img_k, global_step=global_step, **kwargs)

    def add_scalar_dict(self, tag: str, scalar_dict: Dict[str, Any], global_step: int, **kwargs):
        for key, value in scalar_dict.items():
            self.add_scalar(f"{tag}/{key}", value, global_step=global_step)

    def log_normal(self, tag: str, dist_y_hat: torch.distributions.Normal, y: torch.Tensor, global_step: int):
        error = torch.abs(dist_y_hat.mean - y)
        self.add_histogram(f"{tag}/xi", error ** 2 / dist_y_hat.variance, global_step=global_step)
        self.add_histogram(f"{tag}/error", error, global_step=global_step)
        self.add_histogram(f"{tag}/mean", dist_y_hat.mean, global_step=global_step)
        self.add_histogram(f"{tag}/stddev", dist_y_hat.stddev, global_step=global_step)
