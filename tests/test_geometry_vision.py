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


@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.parametrize("num_patches", (1, 4))
@pytest.mark.parametrize("width", (20, 40))
@pytest.mark.parametrize("height", (20, 40))
def test_crop_patches(batch_size, num_patches, width, height):
    patch_size = 4

    images = torch.rand((batch_size, 3, height, width))
    x_center = torch.randint(patch_size, width - patch_size, size=(batch_size, num_patches))
    y_center = torch.randint(patch_size, height - patch_size, size=(batch_size, num_patches))
    center_points = torch.stack((x_center, y_center), dim=-1)
    patches = torram.geometry.crop_patches(images, center_points, width=patch_size, height=patch_size)

    assert patches.shape == (batch_size, num_patches, 3, 2*patch_size, 2*patch_size)
    for b in range(batch_size):
        for n in range(num_patches):
            cx = center_points[b, n, 0]
            cy = center_points[b, n, 1]
            expected = images[b, :, cy-patch_size:cy+patch_size, cx-patch_size:cx+patch_size]
            assert torch.allclose(patches[b, n], expected, atol=1e-3)


def test_crop_patches_outside_image():
    images = torch.rand((4, 3, 20, 20))
    points = torch.randint(25, 35, size=(4, 5, 2))
    patches = torram.geometry.crop_patches(images, points, width=4, height=4)
    assert torch.allclose(patches, torch.zeros_like(patches))


def test_crop_patches_half_in():
    images = torch.rand((1, 3, 20, 20))
    points = torch.tensor([20, 10]).view(1, 1, 2)
    patches = torram.geometry.crop_patches(images, points, width=4, height=4)
    assert torch.all(patches[:, :, 4:, :] == 0)


def test_box_including_2d():
    points = torch.randint(-20, 200, size=(200, 2))
    bbox = torram.geometry.box_including_2d(points, offset=0)
    assert bbox[0] == torch.min(points[:, 0])
    assert bbox[1] == torch.min(points[:, 1])
    assert bbox[2] == torch.max(points[:, 0])
    assert bbox[3] == torch.max(points[:, 1])


def test_box_including_2d_bounds():
    points = torch.tensor([[-2, 3], [3, 7], [0, 120]])
    bbox = torram.geometry.box_including_2d(points, x_max=100, y_max=100, offset=0)
    assert torch.all(torch.less_equal(bbox, 100))


@pytest.mark.parametrize("shape", ((1, ), (4, 2, 1), (1, 1, )))
def test_image_normalization(shape):
    mean = torch.tensor([0.31, 0.1, 0.412]).view(3, 1, 1)
    std = torch.tensor([1.0, 3.1, 2.1]).view(3, 1, 1)
    # un-normalize image by y = (x - mean) / std <=> x = y * std + mean
    images = torch.ones((*shape, 3, 1, 1), dtype=torch.float32) * std + mean
    images = torram.transforms.vision.normalize_images(images, mean=mean, std=std)
    assert torch.allclose(images, torch.ones_like(images))
