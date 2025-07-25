from typing import List, Optional, Tuple, Union

import torch
from jaxtyping import Num

__all__ = ["diag_last", "eye", "eye_like"]


def eye(
    shape: Union[torch.Size, Tuple[int, ...]],
    dtype: Optional[torch.dtype] = None,
    device=None,
    requires_grad: bool = False,
) -> Num[torch.Tensor, "... N N"]:
    """Make identity matrix with given shape (*shape, shape[-1]).

    >>> I = eye((8, 2, 3))
    >>> I.shape
    (8, 2, 3, 3)
    """
    assert len(shape) > 0

    d = shape[-1]
    output = torch.zeros(
        (*shape[:-1], d, d),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )

    for i in range(d):
        output[..., i, i] = 1
    return output


def eye_like(
    x: Num[torch.Tensor, "... N N"], requires_grad: bool = False
) -> Num[torch.Tensor, "... N N"]:
    """Make identity matrix like given matrix (shape, device, dtype).

    >>> y = torch.rand((4, 7, 2, 2))
    >>> I = eye_like(y)
    >>> I.shape
    (4, 7, 2, 2)

    """
    assert x.ndim >= 2 and x.shape[-1] == x.shape[-2]
    return eye(x.shape[:-1], dtype=x.dtype, device=x.device, requires_grad=requires_grad)


def diag_last(x: Num[torch.Tensor, "... N"]) -> Num[torch.Tensor, "... N N"]:
    """Make diagonal matrix from the last dimension of x, for any shape of x."""
    assert x.ndim > 0
    x_eye = eye(x.shape, dtype=x.dtype, device=x.device)
    return x_eye * x.unsqueeze(-1)


def to_device(*args, device: torch.device) -> List[torch.Tensor]:
    return [arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args]


def to_device_dict(d: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}
