import logging
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

    def add_scalar_dicts(self, tag: str, scalar_dict: Dict[str, Any], global_step: int, **kwargs):
        key_lengths = [len(key.split("/")) for key in scalar_dict.keys()]
        if len(key_lengths) == 0:  # empty scalar dict, just return
            return
        if not all(l == key_lengths[0] for l in key_lengths) or key_lengths[0] == 1:
            logging.debug("Not-Matching or unit length keys, fallback to add_scalar_dict")
            self.add_scalar_dict(tag, scalar_dict, global_step=global_step)
            return

        leveled_dict = dict()
        for key, value in scalar_dict.items():
            key_level = tag + "/" + "/".join(key.split("/")[:-1])
            if key_level not in leveled_dict:
                leveled_dict[key_level] = dict()
            key_value = "/".join(key.split("/")[-1:])
            leveled_dict[key_level][key_value] = value
        for key_level, level_dict in leveled_dict.items():
            self.add_scalars(key_level, level_dict, global_step=global_step)

    def add_normal(self, tag: str, mean: torch.Tensor, target: torch.Tensor, variances: torch.Tensor, global_step: int):
        if mean.shape != target.shape:
            raise ValueError(f"Mean and target are not matching, got {mean.shape} and {target.shape}")
        if mean.shape != variances.shape:
            raise ValueError(f"Mean and variances are not matching, got {mean.shape} and {variances.shape}")
        if torch.any(variances < 0):
            raise ValueError(f"Invalid variances, got negative values")
        error = torch.abs(mean - target)
        self.add_histogram(f"{tag}/xi", error ** 2 / variances, global_step=global_step)
        self.add_histogram(f"{tag}/error", error, global_step=global_step)
        self.add_histogram(f"{tag}/mean", mean, global_step=global_step)
        self.add_histogram(f"{tag}/variance", variances, global_step=global_step)
