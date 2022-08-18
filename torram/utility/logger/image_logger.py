import torch
import torram

from pathlib import Path
from .log_logger import LogLogger


class ImageLogger(LogLogger):

    def __init__(self, directory: Path):
        super(ImageLogger, self).__init__()
        self.directory = directory

    def add_image(self, tag: str, img: torch.Tensor, **kwargs):
        output_path = self.directory / f"{tag}.jpg"
        output_path.parent.mkdir(exist_ok=True)
        torram.io.write_jpeg(img, output_path)

    def add_images_w_iter(self, tag: str, images: torch.Tensor, **kwargs):
        for k, img_k in enumerate(images):
            self.add_image(f"{tag}/{k+1}", img_k, **kwargs)
