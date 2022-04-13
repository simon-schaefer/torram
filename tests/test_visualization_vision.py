import pytest
import torch
import torram

from typing import Tuple


@pytest.mark.parametrize("image_shape", ((3, 100, 100), (3, 1000, 200), (3, 2, 2)))
def test_draw_keypoints(image_shape: Tuple[int, int, int]):
    from torchvision.utils import draw_keypoints
    image = torch.zeros(image_shape, dtype=torch.uint8)
    points = torch.stack([torch.randint(0, image_shape[1], size=(4, ), dtype=torch.long),
                          torch.randint(0, image_shape[2], size=(4, ), dtype=torch.long)], dim=1)
    color = (123, 3, 41)

    out_hat = torram.visualization.draw_keypoints(image, points, color=color)
    # The torchvision implementation draws a circle around each keypoint with radius 1. Therefore,
    # all the neighbors have to be colored, too.
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i, j in neighbors:
        points_ij = points.clone()
        points_ij[:, 0] += i
        points_ij[:, 1] += j
        out_hat = torram.visualization.draw_keypoints(out_hat, points_ij, color=color)

    out = draw_keypoints(image, points[None], colors=color, radius=1)
    assert torch.allclose(out_hat, out)


def test_draw_keypoints_outside_points():
    image = torch.zeros((3, 10, 10), dtype=torch.uint8)
    points = torch.tensor([[11, 0], [-1, 1], [-1, -1]], dtype=torch.long)
    out_hat = torram.visualization.draw_keypoints(image, points)
    assert torch.allclose(out_hat, image)
