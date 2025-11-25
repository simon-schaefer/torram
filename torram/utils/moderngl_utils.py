import platform
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import moderngl
import numpy as np
from jaxtyping import Bool, Float, UInt8
from PIL import Image, ImageDraw, ImageFont
from pyrr import Matrix44

from torram.geometry.smpl import SMPL_EDGE_COLORS, SMPL_KINTREE


def draw_transform(
    ctx: moderngl.Context,
    prog,
    transform: Float[np.ndarray, "4 4"],
    axis_length: float = 0.2,
) -> None:
    """Draws a 3D coordinate system at the given transform.

    @param ctx: The moderngl context to use for rendering.
    @param prog: The shader program to use for rendering.
    @param transform: A 4x4 transformation matrix representing the position and orientation.
    @param axis_length: The length of the axes to draw.
    """
    x_axis_B = np.array([1.0, 0.0, 0.0]) * axis_length
    y_axis_B = np.array([0.0, 1.0, 0.0]) * axis_length
    z_axis_B = np.array([0.0, 0.0, 1.0]) * axis_length

    x_axis_W = transform[:3, :3] @ x_axis_B + transform[:3, 3]
    y_axis_W = transform[:3, :3] @ y_axis_B + transform[:3, 3]
    z_axis_W = transform[:3, :3] @ z_axis_B + transform[:3, 3]

    x_color = [1.0, 0.0, 0.0]  # Red
    y_color = [0.0, 1.0, 0.0]  # Green
    z_color = [0.0, 0.0, 1.0]  # Blue

    vertices = [
        *transform[:3, 3].tolist(),
        *x_color,
        *x_axis_W,
        *x_color,
        *transform[:3, 3].tolist(),
        *y_color,
        *y_axis_W,
        *y_color,
        *transform[:3, 3].tolist(),
        *z_color,
        *z_axis_W,
        *z_color,
    ]
    vertices = np.array(vertices, dtype="f4")

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_color")])
    vao.render(mode=moderngl.LINES)


def draw_path(
    ctx: moderngl.Context,
    prog,
    path: Float[np.ndarray, "N 3"],
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Draws a path in 3D space.

    @param ctx: The moderngl context to use for rendering.
    @param prog: The shader program to use for rendering.
    @param path: An array of shape (N, 3) representing the path points.
    @param color: A tuple representing the RGB color of the path.
    """
    if len(path) < 2:
        return  # No path to draw

    vertices = np.zeros((len(path) * 4, 3), dtype="f4")
    vertices[::4, :] = path
    vertices[1::4, :] = color
    vertices[2::4, :] = np.concatenate((path[1:], path[-1:]), axis=0)
    vertices[3::4, :] = color

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_color")])
    vao.render(mode=moderngl.LINES)


def draw_smpl_skeleton(ctx: moderngl.Context, prog, joints: Float[np.ndarray, "N 3"]):
    """Draws a skeleton from SMPL joints.

    @param ctx: The moderngl context to use for rendering.
    @param prog: The shader program to use for rendering.
    @param joints: An array of shape (N, 3) representing the joint positions.
    """
    draw_skeleton(ctx, prog, joints, SMPL_KINTREE, SMPL_EDGE_COLORS)


def draw_skeleton(
    ctx: moderngl.Context,
    prog,
    joints: Float[np.ndarray, "N 3"],
    kintree: List[Tuple[int, int]],
    colors: List[Tuple[float, float, float]],
    mask: Optional[Bool[np.ndarray, "N"]] = None,
):
    """Draws a skeleton from joint positions and a kinematic tree.

    @param ctx: The moderngl context to use for rendering.
    @param prog: The shader program to use for rendering.
    @param joints: An array of shape (N, 3) representing the joint positions.
    @param kintree: A list of tuples representing the kinematic tree edges.
    @param colors: A list of RGB tuples representing the colors for each edge.
    @param mask: An optional boolean array to mask out certain joints.
    """
    assert len(kintree) == len(colors), "Kinematic tree and colors must have the same length"
    num_edges = len(kintree)

    vertices = np.zeros((num_edges * 4, 3), dtype="f4")
    for i, (start, end) in enumerate(kintree):
        if mask is not None and (not mask[start] or not mask[end]):
            continue
        vertices[i * 4 + 0] = joints[start]
        vertices[i * 4 + 1] = colors[i]
        vertices[i * 4 + 2] = joints[end]
        vertices[i * 4 + 3] = colors[i]
    vertices = vertices.flatten()

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_color")])
    vao.render(mode=moderngl.LINES)


def draw_point(
    ctx: moderngl.Context,
    prog,
    position: Float[np.ndarray, "3"],
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Draws a point in 3D space.

    @param ctx: The moderngl context to use for rendering.
    @param prog: The shader program to use for rendering.
    @param position: A 3D position of the point.
    @param color: A tuple representing the RGB color of the point.
    """
    vertices = np.array([*position, *color], dtype="f4")
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "in_position", "in_color")])
    vao.render(mode=moderngl.POINTS, vertices=1)


def draw(
    draw_frame: Callable,
    num_frames: int,
    caption: Optional[str] = None,
    window_size: Tuple[int, int] = (256, 256),
    camera_view: Union[Literal["top"], Matrix44, Callable] = "top",
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> UInt8[np.ndarray, "N H W C"]:
    aspect_ratio = window_size[0] / window_size[1]

    system = platform.system().lower()
    if system == "darwin":  # macOS
        ctx = moderngl.create_standalone_context()
    else:  # Linux / Windows
        ctx = moderngl.create_standalone_context(backend="egl")

    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    fbo = ctx.simple_framebuffer(window_size)
    fbo.use()
    prog = ctx.program(
        vertex_shader="""
            #version 330
            uniform mat4 model;
            uniform mat4 projection;
            uniform mat4 view;
            in vec3 in_position;
            in vec3 in_color;
            out vec3 color;
            void main() {
                gl_Position = projection * view * model * vec4(in_position, 1.0);
                color = in_color;
                gl_PointSize = 8.0;
            }
        """,
        fragment_shader="""
            #version 330
            in vec3 color;
            out vec4 fragColor;
            void main() {
                fragColor = vec4(color, 1.0);
            }
        """,
    )

    projection = Matrix44.perspective_projection(60.0, aspect_ratio, 0.1, 100.0)
    if isinstance(camera_view, Matrix44):
        view = camera_view
    elif isinstance(camera_view, str) and camera_view == "top":
        view = Matrix44.look_at(
            eye=(0.0, 0.0, 10.0),
            target=(0.0, 0.0, 1.0),
            up=(0.0, 1.0, 0.0),
        )
    else:
        raise ValueError(f"Unsupported camera view: {camera_view}")
    prog["model"].write(np.eye(4, dtype="f4").tobytes())
    prog["view"].write(view.astype("f4").tobytes())
    prog["projection"].write(projection.astype("f4").tobytes())

    frames = np.zeros((num_frames, window_size[1], window_size[0], 3), dtype=np.uint8)
    for k in range(num_frames):
        ctx.clear(red=background_color[0], green=background_color[1], blue=background_color[2])

        draw_frame(ctx, prog, k)

        data = fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(window_size[1], window_size[0], 3)
        img = np.rot90(img, k=-1)

        if caption is not None:
            font = ImageFont.load_default(size=20)
            img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), caption, font=font, fill=(0, 0, 0))
            img = np.array(img)

        frames[k] = img

    fbo.release()
    ctx.release()
    prog.release()

    return frames


def draw_batch(
    draw_frame: Callable,
    batch_size: int,
    num_frames: int,
    caption: Optional[str] = None,
    window_size: Tuple[int, int] = (256, 256),
    camera_view: Union[Literal["top"], Matrix44, Callable] = "top",
    background_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    output_format: Literal["BNCHW", "BNHWC"] = "BNCHW",
) -> UInt8[np.ndarray, "B N H W C"]:
    output = np.zeros((batch_size, num_frames, window_size[1], window_size[0], 3), dtype=np.uint8)
    for i in range(batch_size):
        output[i] = draw(
            draw_frame=partial(draw_frame, batch_index=i),
            num_frames=num_frames,
            caption=caption,
            window_size=window_size,
            camera_view=camera_view if not isinstance(camera_view, Callable) else camera_view(i),
            background_color=background_color,
        )

    if output_format == "BNCHW":
        output = np.permute_dims(output, (0, 1, 4, 2, 3))
    elif output_format == "BNHWC":
        pass  # Already in the desired output format
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return output
