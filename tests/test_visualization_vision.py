import os
import torch
import torram


def test_draw_keypoints_outside_points():
    image = torch.zeros((3, 10, 10), dtype=torch.uint8)
    points = torch.tensor([[11, 0], [-2, 1], [-1, -1]], dtype=torch.long)
    out_hat = torram.visualization.draw_keypoints(image, points[None], radius=1)
    assert torch.allclose(out_hat, image)


def test_draw_reprojection():
    image = torch.zeros((3, 480, 640), dtype=torch.uint8)
    _, h, w = image.shape
    points_3d = torch.rand((10, 3)) * 3
    points_3d[:, 2] = 10
    K = torch.tensor([[384, 0, w/2],
                      [0, 384, h/2],
                      [0, 0, 1]], dtype=torch.float32)

    points_2d = torram.geometry.project_points(points_3d, camera_matrix=K).long()
    image_expected = torram.visualization.draw_keypoints(image, points_2d[None], radius=3, colors="red")
    image_hat = torram.visualization.draw_reprojection(image, points_3d[None], K=K, radius=3, colors="red")
    assert torch.allclose(image_hat, image_expected)


def test_draw_keypoints_weighted():
    image_shape = (3, 100, 100)
    cache_directory = os.environ.get("CACHE_DIRECTORY", "/srv/cache")
    output_directory = os.path.join(cache_directory, "keypoints_weighted")
    os.makedirs(output_directory, exist_ok=True)

    image = torch.zeros(image_shape, dtype=torch.uint8)
    points = torch.stack([torch.randint(0, image_shape[1], size=(20, ), dtype=torch.long),
                          torch.randint(0, image_shape[2], size=(20, ), dtype=torch.long)], dim=1)
    scores = torch.rand((20, ), dtype=torch.float32)

    out = torram.visualization.draw_keypoints_weighted(image, points, scores=scores)
    assert out.shape == (3, 100, 100)
    torram.io.write_png(out, os.path.join(output_directory, f"test.png"))
