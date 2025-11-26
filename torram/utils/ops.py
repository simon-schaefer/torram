from typing import Dict, List, Optional, Tuple, Union

import numpy as np
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
    return [
        arg.to(device) if (isinstance(arg, torch.Tensor) or hasattr(args, "to")) else arg
        for arg in args
    ]


def to_device_dict(d: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if (isinstance(v, torch.Tensor) or hasattr(v, "to")) else v
        for k, v in d.items()
    }


def stack_nested(xs):
    """Stack a list of nested dicts of tensors / arrays.

    Example:

    a = {"level1": {"level2": {"x": torch.tensor([1,2]), "y": torch.tensor([3])},
                    "other": torch.tensor([10])}, "meta": "a"}
    b = {"level1": {"level2": {"x": torch.tensor([4,5]), "y": torch.tensor([6])},
                    "other": torch.tensor([20])}, "meta": "b"}
    stacked = stack_nested([a, b])

    Output:

    stacked = {
      "level1": {
        "level2": {"x": (2,2), "y": (2,1)},
        "other": (2,1)
      },
      "meta": ["a", "b"]
    }
    """
    if not isinstance(xs[0], dict):
        if isinstance(xs[0], torch.Tensor):
            return torch.stack(xs, dim=0)
        elif isinstance(xs[0], np.ndarray):
            return np.stack(xs, axis=0)
        else:
            return xs  # non-tensor leaf → collected into list

    out = {}
    for key in xs[0].keys():
        out[key] = stack_nested([x[key] for x in xs])
    return out


def slice_stacked(struct, a: int, b: int, copy: bool = False):
    """Recursively slice a stacked nested dict of tensors / arrays from index a to b.

    @param struct: Nested structure (dict, list, tuple) of tensors / arrays.
    @param a: Start index.
    @param b: End index.
    @return: Sliced nested structure.
    """
    if isinstance(struct, dict):
        return {k: slice_stacked(v, a, b) for k, v in struct.items()}
    if isinstance(struct, torch.Tensor):
        return struct[a:b].clone()
    if isinstance(struct, np.ndarray):
        return struct[a:b].copy()
    if isinstance(struct, (list, tuple)):
        sliced = struct[a:b]
        return sliced if isinstance(struct, list) else tuple(sliced)

    # Anything else treat as scalar/leaf → return as-is
    return struct


def collect_nested_leaf_lengths(x):
    """Recursively collect all leaf lengths from a nested structure."""
    if isinstance(x, dict):
        lengths = []
        for v in x.values():
            lengths.extend(collect_nested_leaf_lengths(v))
        return lengths
    else:
        return [len(x)]
