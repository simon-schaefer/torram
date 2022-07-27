import torch

from typing import Tuple, Union

__all__ = [
    'diag_last',
    'eye',
    'eye_like'
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


def eye_like(x: torch.Tensor, requires_grad = False) -> torch.Tensor:
    """Make identity matrix like given matrix (shape, device, dtype).

    >>> y = torch.rand((4, 7, 2, 2))
    >>> I = eye_like(y)
    >>> I.shape
    (4, 7, 2, 2)

    Args:
        x: input matrix to mimic (..., a, a).
        requires_grad: required gradient for created identity tensor.
    Returns:
        identity matrix similar to input matrix in shape, device and dtype.
    """
    if len(x.shape) < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError("Invalid input matrix, must be at least two-dimensional and square")
    return eye(x.shape[:-1], dtype=x.dtype, device=x.device, requires_grad=requires_grad)
