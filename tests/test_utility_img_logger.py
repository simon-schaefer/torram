import os
import torch
import torram


def test_add_image():
    cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "cache")
    logger = torram.utility.logger.ImageLogger(cache_dir)
    image = torch.zeros((3, 400, 600), dtype=torch.uint8)
    logger.add_image("img_logger_add_image", image)


def test_no_error_other_function():
    cache_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "cache")
    logger = torram.utility.logger.ImageLogger(cache_dir)
    logger.add_histogram("test", None, global_step=5)
