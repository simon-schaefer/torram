import torch

from typing import Tuple, Union

__all__ = [
    'diag_last',
    'eye'
]


def diag_last(x: torch.Tensor) -> torch.Tensor:
    """"Make diagonal matrix from the last dimension of x, for any shape of x.

    Args:
        x: input tensor (..., D).
    Returns:
        diagonal tensor with the last dimension of x as its diagonal.
    """
    if len(x.shape) == 0:
        raise ValueError("Cannot make a diagonal matrix with empty tensor")
    d = x.shape[-1]

    output = torch.zeros((*x.shape[:-1], d, d), dtype=x.dtype, device=x.device)
    for i in range(d):
        output[..., i, i] = x[..., i]
    return output


def eye(shape: Union[torch.Size, Tuple[int, ...]], dtype: torch.dtype = None, device=None, requires_grad: bool = False
        ) -> torch.Tensor:
    """Make identity matrix with given shape (*shape, shape[-1]).

    >>> I = eye((8, 2, 3))
    >>> I.shape
    (8, 2, 3, 3)
    """
    if len(shape) == 0:
        raise ValueError(f"Got empty shape for eye")
    out_diagonal = torch.ones(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return diag_last(out_diagonal)
