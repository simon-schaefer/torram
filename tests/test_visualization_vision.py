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
