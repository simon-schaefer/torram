import os
import pathlib
import torch
import torram


def test_add_image():
    cache_dir = pathlib.Path(os.path.realpath(__file__)).parent / "assets" / "cache"
    logger = torram.utility.logger.ImageLogger(cache_dir)
    image = torch.zeros((3, 400, 600), dtype=torch.uint8)
    logger.add_image("img_logger_add_image", image)


def test_no_error_other_function():
    cache_dir = pathlib.Path(os.path.realpath(__file__)).parent / "assets" / "cache"
    logger = torram.utility.logger.ImageLogger(cache_dir)
    logger.add_histogram("test", None, global_step=5)
