import torch
import torram.metrics

__all__ = ['mpjpe',
           'pa_mpjpe']


def mpjpe(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error after pelvis alignment.

    Args:
        x_hat: predicted joints (B, N_joints, 3).
        x: ground-truth joints (B, N_joints, 3).
    """
    if not x_hat.ndim == x.ndim == 3:
        raise ValueError(f"Invalid joint shape, expected (B, N, 3), got {x.shape} and {x_hat.shape}")
    x_hat_aligned = __align_by_pelvis(x_hat)
    x_aligned = __align_by_pelvis(x)
    return torram.metrics.pve(x_hat_aligned, x_aligned)


def pa_mpjpe(x_hat: torch.Tensor, x: torch.Tensor, align_scale: bool = True) -> torch.Tensor:
    """Mean Per Joint Position Error after Procrustes alignment. Inspired by
    https://github.com/akanazawa/human_dynamics/blob/0887f37464c9a079ad7d69c8358cecd0f43c4f2a/src/evaluation/

    Args:
        x_hat: predicted joints (B, N_joints, 3).
        x: ground-truth joints (B, N_joints, 3).
        align_scale: align scale or only rotation and translation?
    """
    x_hat_aligned = __align_by_pelvis(x_hat)
    x_aligned = __align_by_pelvis(x)
    return torram.metrics.pa_pve(x_hat_aligned, x_aligned, align_scale=align_scale)


def __align_by_pelvis(x: torch.Tensor, left_id: int = 2, right_id: int = 1):
    """Aligns joints by pelvis to be at origin. Assumes that the pelvis is the 1st and 2nd joint, according to
    https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
    """
    pelvis = (x[..., left_id, :] + x[..., right_id, :]) / 2.  # midpoint
    return x - pelvis.unsqueeze(dim=-2)  # translate joints to origin based on pelvis
