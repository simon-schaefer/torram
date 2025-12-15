from typing import List, Optional, Tuple, Union

import numpy as np
import rerun as rr
import trimesh
from jaxtyping import Bool, Float

from torram.geometry.smpl import SMPL_EDGE_COLORS, SMPL_KINTREE

NodeColorsType = Union[
    Float[np.ndarray, "J 3"],
    Float[np.ndarray, "3"],
    Tuple[float, float, float],
    None,
]

EdgeColorsType = Union[
    Float[np.ndarray, "E 3"],
    Float[np.ndarray, "3"],
    Tuple[float, float, float],
    None,
]


def log_body_skeleton(
    tag: str,
    joints: Float[np.ndarray, "J 3"],
    connections: List[Tuple[int, int]],
    mask: Optional[Bool[np.ndarray, "J"]] = None,
    *,
    radius: float = 0.03,
    colors: NodeColorsType = None,
    edge_colors: EdgeColorsType = None,
) -> None:
    """
    Log a body skeleton in rerun.

    @param tag: Entity path to log the skeleton to.
    @param joints: 3D joint positions.
    @param connections: List of joint index pairs defining the skeleton connections.
    @param mask: Optional boolean mask indicating valid joints.
    @param radius: Radius of the joint spheres.
    @param colors: Optional colors for the joints (as RGB values in [0, 1]).
    @param edge_colors: Optional colors for the skeleton edges (as RGB values in [0, 1]).
    """
    if mask is not None:
        joints_masked = joints[mask]
        edges = []
        for a, b in connections:
            valid = mask[a] & mask[b]
            edges.append(joints[valid][:, [a, b]])
        edges = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2, 3))
    else:
        joints_masked = joints
        edges = joints[connections]

    rr.log(f"{tag}/joints", rr.Points3D(joints_masked, radii=radius, colors=colors))
    rr.log(f"{tag}/skeleton", rr.LineStrips3D(edges, colors=edge_colors))


def log_smpl_skeleton(
    tag: str,
    joints: Float[np.ndarray, "J 3"],
    mask: Optional[Bool[np.ndarray, "J"]] = None,
    *,
    radius: float = 0.03,
    colors: NodeColorsType = None,
    edge_colors: EdgeColorsType = None,
) -> None:
    """
    Log an SMPL body skeleton in rerun.

    @param tag: Entity path to log the skeleton to.
    @param joints: 3D joint positions.
    @param mask: Optional boolean mask indicating valid joints.
    @param radius: Radius of the joint spheres.
    @param colors: Optional colors for the joints (as RGB values in [0, 1]).
    @param edge_colors: Optional colors for the skeleton edges (as RGB values in [0, 1]).
    """
    if edge_colors is None:
        edge_colors = np.array(SMPL_EDGE_COLORS)
    if colors is None:
        colors = np.ones((joints.shape[0], 3)) * 0.5

    log_body_skeleton(
        tag,
        joints,
        connections=SMPL_KINTREE,
        mask=mask,
        radius=radius,
        edge_colors=edge_colors,
        colors=colors,
    )


def log_trimesh(
    tag: str,
    mesh: trimesh.Trimesh,
    transform: Optional[np.ndarray] = None,
    static: bool = False,
) -> None:
    """
    Log a trimesh mesh in rerun.

    @param tag: Entity path to log the mesh to.
    @param mesh: Trimesh mesh to log.
    @param transform: Optional 4x4 transformation matrix to apply to the mesh before logging.
    @param static: Whether the mesh is static (does not change over time).
    """
    if transform is not None:
        mesh = mesh.copy()
        mesh.apply_transform(transform)

    if mesh.visual is not None and mesh.visual.kind != "vertex":
        vertex_colors = mesh.visual.vertex_colors
        vertex_colors = vertex_colors[:, :3] if vertex_colors is not None else None
    else:
        vertex_colors = None

    rr.log(
        tag,
        rr.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=vertex_colors,
        ),
        static=static,
    )


def log_transform(
    tag: str,
    transform: Float[np.ndarray, "4 4"],
    static: bool = False,
    axis_length: float = 0.1,
) -> None:
    """
    Log a 3D transformation in rerun.

    @param tag: Entity path to log the transform to.
    @param transform: 4x4 transformation matrix.
    @param static: Whether the transform is static (does not change over time).
    @param axis_length: Length of the axes to visualize.
    """
    t, R = transform[:3, 3], transform[:3, :3]
    rr.log(
        tag,
        rr.Transform3D(translation=t, mat3x3=R, axis_length=axis_length),
        static=static,
    )
