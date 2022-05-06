import logging
import torch
import torram
from typing import Tuple

__all__ = ['occupancy_grid_from_point_cloud']


def occupancy_grid_from_point_cloud(point_cloud: torch.Tensor, T_B_C: torch.Tensor,
                                    grid_size: Tuple[float, float, float], voxel_size: float = 0.2) -> torch.Tensor:
    """Compute occupancy grid based on a point cloud by filling every voxel in a voxel grid which contains
    at least one point of the point cloud.

    Args:
        point_cloud: point cloud to fill occupancy map [B, N, 3].
        T_B_C: transform from point_cloud coordinate system to body coordinate systems [B, M, 4, 4].
        grid_size: number of voxels in each direction, x, y, z.
        voxel_size: size of each voxel in meters, same in each direction.
    """
    if len(T_B_C.shape) != 4 or T_B_C.shape[-1] != 4 or T_B_C.shape[-2] != 4:
        raise ValueError(f"Input transformation must have shape (B, M, 4, 4). Got {T_B_C.shape}")
    if len(point_cloud.shape) != 3 or point_cloud.shape[-1] != 3:
        raise ValueError(f"Input point cloud must have shape (B, N, 3). Got {point_cloud.shape}")
    if T_B_C.shape[0] != point_cloud.shape[0]:
        raise ValueError(f"Input point cloud and transformation batch size must match, "
                         f"got {point_cloud.shape} and {T_B_C.shape}")

    logging.debug(f"Creating a voxel grid with size {grid_size} and voxel size {voxel_size}")
    nx = int(grid_size[0] / voxel_size)
    ny = int(grid_size[1] / voxel_size)
    nz = int(grid_size[2] / voxel_size)
    batch_size, num_bodies, _, _ = T_B_C.shape
    device = point_cloud.device
    occupancy_grid = torch.zeros((batch_size, num_bodies, nx, ny, nz), dtype=torch.bool, device=device)

    logging.debug(f"... transforming points to {num_bodies} body coordinate system")
    point_cloud_homo = torram.geometry.convert_points_to_homogeneous(point_cloud)
    T_G_B = torch.eye(4, device=device, dtype=point_cloud.dtype)
    T_G_B[0, 3] = grid_size[0] / 2
    T_G_B[1, 3] = grid_size[1] / 2
    T_G_B[2, 3] = grid_size[2] / 2
    T_G_B = T_G_B.view(1, 4, 4).repeat(batch_size, 1, 1)
    T_G_C = torch.stack([torch.bmm(T_G_B, T_B_C[:, m]) for m in range(num_bodies)], dim=1)
    point_cloud_B = torch.stack([torch.bmm(T_G_C[:, m], point_cloud_homo.transpose(1, 2))
                                 for m in range(num_bodies)], dim=1).transpose(2, 3)[..., :3]

    logging.debug(f"... filling voxel grid with {point_cloud_B.shape[-2]} points")
    pc_index_B = torch.div(point_cloud_B, voxel_size, rounding_mode='floor').long()

    logging.debug("... removing invalid points, outside of the voxel grid")
    lower_in = torch.less(pc_index_B, torch.tensor([nx, ny, nz], device=device, dtype=torch.long))
    upper_in = torch.greater_equal(pc_index_B, 0)
    in_bounds = torch.all(torch.logical_and(lower_in, upper_in), dim=-1)  # over all axes (x, y, z)

    logging.debug(f"... inserting {pc_index_B.shape[-2]} points as occupancies in voxel grid")
    for b in range(batch_size):
        for m in range(num_bodies):
            pc_index_bm = pc_index_B[b, m, in_bounds[b, m]]
            occupancy_grid[b, m, pc_index_bm[:, 0], pc_index_bm[:, 1], pc_index_bm[:, 2]] = True
    return occupancy_grid
