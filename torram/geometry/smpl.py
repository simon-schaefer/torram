from typing import Dict

import smplx
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle


def transform_body_parameters(
    body_params: Dict[str, torch.Tensor],
    transform: torch.Tensor,
    smpl_model: smplx.SMPL,
) -> Dict[str, torch.Tensor]:
    """Transforms SMPL body parameters by a given rigid transformation.

    Let assume that you have estimated some SMPL/SMPL+H/SMPL-X parameters from an image using the method of your choice.
    At the same time, you somehow have the 3D structure of the scene and you know the rigid transformation (R_w, t_w)
    that transforms point in the camera  coordinate frame to the world coordinate frame.

    Now you want to get SMPL/SMPL+H/SMPL-X that directly pose your model in the world coordinate frame.
    How do you do this?

    1. First, the center of rotation of your models is the PELVIS . The pelvis location changes according to the shape.
        1. Let $$p = p(\beta)$$ be the pelvis location for your estimated shape parameters.

    2. The vertices in the camera coordinates are:
        $$v_c = R_c (v - p) + p + t_c$$ (1)
        where $$v$$ are the posed vertices, before applying the global orientation, $$R_c$$ is the
        global orientation in the camera coordinate frame, $$t_c$$ is the translation in the camera coordinate frame.

    3. The vertices in the world coordinate frame are:
        $$v_w = R_c^w v_c + t_c^w$$ (2)

    4. What we want is to estimate $$(R_w, t_w)$$ so that we can directly pose our model in the world coordinate frame,
    without this two stage process. In other words:
        $$v_w = R_w (v - p) + p + t_w$$ (3)

    5. So let’s replace $$v_c$$ in eq. 2 with eq. 1 and make this equal to eq. 3:
        $$R_w (v - p) + p + t_w = R_c^w (R_c (v - p) + p + t_c) + t_c^w$$
        $$R_w (v - p) + p + t_w = R_c^w R_c (v - p) + p - p +  R_c^w (p + t_c) + t_c^w$$
        $$R_w (v - p) + p + t_w = R_c^w R_c (v - p) + p + (R_c^w (p + t_c) + t_c^w - p)$$

    6. So, in the end: The new global orientation is the composition of the camera-to-world
    transformation and the old global orientation
        $$R_w = R_c^w R_c$$ :
        $$t_w =(R_c^w (p + t_c) + t_c^w - p)$$

    7. Now you can directly pose your model in the new coordinate system!

    Source: https://www.dropbox.com/scl/fi/zkatuv5shs8d4tlwr8ecc/Change-parameters-to-new-coordinate-system.paper?rlkey=lotq1sh6wzkmyttisc05h0in0&dl=0

    @params body_params: Dictionary of SMPL body parameters.
    @params transform: Rigid transformation matrix
    @params smpl_model: SMPL model to use for joint computation.
    @return: Transformed body parameters.
    """
    betas = body_params["betas"]
    q = body_params["global_orient"]
    transl = body_params["transl"]

    body_params_zero = {k: v for k, v in body_params.items() if k not in ["betas", "transl"]}
    body_params_zero["betas"] = torch.zeros_like(betas)
    body_params_zero["transl"] = torch.zeros_like(transl)
    joints3d_q0 = smpl_model(**body_params_zero).joints
    p = joints3d_q0[:, 0, :]

    Rq = axis_angle_to_rotation_matrix(q)
    R = transform[:, :3, :3]
    t = transform[:, :3, 3]
    R_out = R @ Rq
    t_out = torch.einsum("bij,bj->bi", R, p + transl) + t - p

    body_params_out = body_params.copy()
    body_params_out["global_orient"] = rotation_matrix_to_axis_angle(R_out)
    body_params_out["transl"] = t_out
    return body_params_out
