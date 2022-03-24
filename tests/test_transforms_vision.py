import torch
import torram
import pytest


@pytest.mark.parametrize("shape", ((1, ), (4, 2, 1), (1, 1, )))
def test_image_normalization(shape):
    mean = torch.tensor([0.31, 0.1, 0.412]).view(3, 1, 1)
    std = torch.tensor([1.0, 3.1, 2.1]).view(3, 1, 1)
    # un-normalize image by y = (x - mean) / std <=> x = y * std + mean
    images = torch.ones((*shape, 3, 1, 1), dtype=torch.float32) * std + mean
    images = torram.transforms.vision.normalize_images(images, mean=mean, std=std)
    assert torch.allclose(images, torch.ones_like(images))


def test_image_normalization_uint8():
    mean = torch.tensor([0.1, 0.1, 0.1])
    std = torch.tensor([0.1, 0.1, 0.1])
    images = torch.ones((1, 3, 1, 1), dtype=torch.uint8) * 255
    images = torram.transforms.vision.normalize_images(images, mean=mean, std=std)

    expected = (torch.ones_like(images, dtype=torch.float32) - mean) / std
    assert images.dtype == torch.float32
    assert torch.allclose(images, expected)
