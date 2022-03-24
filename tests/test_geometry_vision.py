import pytest
import torch
import torram


@pytest.mark.parametrize("shape", ((4, 5, 3), (1, ), (1, 1, 1)))
def test_pad(shape):
    images = torch.rand((*shape, 400, 600), dtype=torch.float32)
    K = torch.zeros((3, 3), dtype=torch.float32)
    K[0, 0] = 300  # random value
    K[1, 1] = 313  # random value
    K[0, 2] = 600 / 2
    K[1, 2] = 400 / 2

    images_pad, K_pad = torram.geometry.pad(images, K, output_shape=(1024, 960))

    assert images_pad.shape[:-2] == images.shape[:-2]
    assert images_pad.shape[-1] == 960
    assert images_pad.shape[-2] == 1024

    assert K_pad.shape == K.shape
    expected = torch.ones_like(K_pad[..., 0, 2]) * 960 / 2
    assert torch.allclose(K_pad[..., 0, 2], expected, atol=20)   # image center mostly directly in center
    expected = torch.ones_like(K_pad[..., 0, 2]) * 1024 / 2
    assert torch.allclose(K_pad[..., 1, 2], expected, atol=20)


@pytest.mark.parametrize("shape", ((4, 5, 3), (1, ), (1, 1, 1)))
def test_is_in_image_none(shape):
    pixel = torch.ones((*shape, 2)) * 10
    is_in_image = torram.geometry.is_in_image(pixel, width=5, height=5)
    assert not torch.all(is_in_image)


@pytest.mark.parametrize("shape", ((4, 5, 3), (1, ), (1, 1, 1)))
def test_is_in_image_only_one(shape):
    pixel = torch.ones((*shape, 2)) * 10
    pixel[..., 0, :] = 1
    is_in_image = torram.geometry.is_in_image(pixel, width=5, height=5)
    assert torch.all(is_in_image[..., 0])
