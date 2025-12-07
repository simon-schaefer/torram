from typing import Dict, Tuple

import smplx
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

SMPL_KINTREE = [
    (0, 1),  # pelvis -> left_hip
    (0, 2),  # pelvis -> right_hip
    (0, 3),  # pelvis -> spine1
    (1, 4),  # left_hip -> left_knee
    (2, 5),  # right_hip -> right_knee
    (3, 6),  # spine1 -> spine2
    (4, 7),  # left_knee -> left_ankle
    (5, 8),  # right_knee -> right_ankle
    (6, 9),  # spine2 -> spine3
    (7, 10),  # left_ankle -> left_foot
    (8, 11),  # right_ankle -> right_foot
    (9, 12),  # spine3 -> neck
    (12, 15),  # neck -> head
    (9, 13),  # spine3 -> left_shoulder
    (9, 14),  # spine3 -> right shoulder
    (13, 16),  # left_shoulder -> left_elbow
    (14, 17),  # right_shoulder -> right_elbow
    (16, 18),  # left_elbow -> left_wrist
    (17, 19),  # right_elbow -> right_wrist
    (18, 20),  # left_wrist -> left_hand
    (19, 21),  # right_wrist -> right_hand
]

SMPL_EDGE_COLORS = [
    (1.0, 0.0, 0.0),  # pelvis -> left_hip
    (0.0, 1.0, 0.0),  # pelvis -> right_hip
    (0.0, 0.0, 1.0),  # pelvis -> spine1
    (1.0, 0.5, 0.0),  # left_hip -> left_knee
    (0.0, 0.5, 1.0),  # right_hip -> right_knee
    (0.5, 0.0, 1.0),  # spine1 -> spine2
    (1.0, 1.0, 0.0),  # left_knee -> left_ankle
    (0.0, 1.0, 1.0),  # right_knee -> right_ankle
    (1.0, 0.0, 1.0),  # spine2 -> spine3
    (0.5, 0.5, 0.0),  # left_ankle -> left_foot
    (0.0, 0.5, 0.5),  # right_ankle -> right_foot
    (0.5, 0.0, 0.5),  # spine3 -> neck
    (0.25, 0.25, 0.25),  # neck -> head
    (0.75, 0.0, 0.75),  # spine3 -> left_shoulder
    (0.0, 0.75, 0.75),  # spine3 -> right shoulder
    (0.75, 0.75, 0.0),  # left_shoulder -> left_elbow
    (0.25, 0.75, 0.25),  # right_shoulder -> right_elbow
    (0.75, 0.25, 0.25),  # left_elbow -> left_wrist
    (0.25, 0.25, 0.75),  # right_elbow -> right_wrist
    (0.5, 0.25, 0.75),  # left_wrist -> left_hand
    (0.75, 0.25, 0.5),  # right_wrist -> right_hand
]

SMPL_JOINT_NAMES = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}


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


def complete_smplx_params(
    smplx_params: Dict[str, torch.Tensor],
    base_shape: Tuple[int, ...],
    add_masks: bool = False,
) -> Dict[str, torch.Tensor]:
    """Completes missing SMPL-X parameters with zeros.

    @params smplx_params: Dictionary of SMPL-X body parameters.
    @params base_shape: Base shape of the parameters (excluding the last dimension).
    @params add_masks: Whether to add masks for each parameter, indicating which parameters were originally present.
    @return: Completed body parameters.
    """
    assert len(smplx_params) > 0, "SMPL-X parameters dictionary is empty."

    for key in [
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "left_hand_pose",
        "right_hand_pose",
        "expression",
        "betas",
        "global_orient",
        "body_pose",
        "transl",
    ]:
        key_mask = f"{key}_mask"
        if key in smplx_params:
            shape = smplx_params[key].shape
            assert (
                shape[:-1] == base_shape
            ), f"SMPL-X parameter '{key}' has incompatible shape {shape} with base shape {base_shape}."
            if add_masks:
                dim = smplx_params[key].shape[-1]
                smplx_params[key_mask] = torch.ones((*base_shape, dim), dtype=torch.bool)
            continue

        if key in ["left_hand_pose", "right_hand_pose"]:
            dim = 45
        elif key == "body_pose":
            dim = 63
        elif key in ["jaw_pose", "leye_pose", "reye_pose"]:
            dim = 3
        elif key in ["transl", "global_orient"]:
            dim = 3
        else:
            dim = 10
        smplx_params[key] = torch.zeros((*base_shape, dim), dtype=torch.float32)
        if add_masks:
            smplx_params[key_mask] = torch.zeros((*base_shape, dim), dtype=torch.bool)

    return smplx_params
