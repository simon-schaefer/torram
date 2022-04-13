import os
import torch
import torram

from .log_logger import LogLogger


class ImageLogger(LogLogger):

    def __init__(self, directory: str):
        super(ImageLogger, self).__init__()
        self.directory = directory

    def add_image(self, tag: str, img: torch.Tensor, **kwargs):
        output_path = os.path.join(self.directory, f"{tag}.jpg")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torram.io.write_jpeg(img, output_path)

    def add_images_w_iter(self, tag: str, images: torch.Tensor, **kwargs):
        for k, img_k in enumerate(images):
            self.add_image(f"{tag}/{k+1}", img_k, **kwargs)
