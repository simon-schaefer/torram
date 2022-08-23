import torch
import torram


def test_occupancy_grid_from_point_cloud_all_outside():
    pc_xyz = 2.0
    grid_size = (2, 2, 2)  # (from -1 to 1 with voxel_size = 1 and center at (0, 0, 0))
    pc = torch.ones((1, 20, 3)) * pc_xyz
    T = torch.eye(4)[None, None].repeat(1, 3, 1, 1)
    grid = torram.estimation.occupancy_grid_from_point_cloud(pc, T, grid_size=grid_size, voxel_size=1)
    assert not torch.all(grid.flatten())
    assert grid.shape == torch.Size([1, 3, 2, 2, 2])


def test_occupancy_grid_from_point_cloud_only_center():
    pc = torch.zeros((1, 20, 3))
    T = torch.eye(4)[None, None]
    grid = torram.estimation.occupancy_grid_from_point_cloud(pc, T, grid_size=(3, 3, 3), voxel_size=1)
    assert grid.shape == torch.Size([1, 1, 3, 3, 3])
    assert grid.sum() == 1  # only center cell is True
    assert torch.all(grid[..., 1, 1, 1])
