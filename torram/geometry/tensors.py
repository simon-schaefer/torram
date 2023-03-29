import torch

from typing import Tuple, Union

__all__ = [
    'diag_last',
    'eye',
    'eye_like'
]


def eye(shape: Union[torch.Size, Tuple[int, ...]], dtype: torch.dtype = None, device=None, requires_grad: bool = False
        ) -> torch.Tensor:
    """Make identity matrix with given shape (*shape, shape[-1]).

    >>> I = eye((8, 2, 3))
    >>> I.shape
    (8, 2, 3, 3)
    """
    assert len(shape) > 0
    d = shape[-1]
    output = torch.zeros((*shape[:-1], d, d), dtype=dtype, device=device, requires_grad=requires_grad)
    for i in range(d):
        output[..., i, i] = 1
    return output


def eye_like(x: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
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
    assert x.ndim >= 2 and x.shape[-1] == x.shape[-2]
    return eye(x.shape[:-1], dtype=x.dtype, device=x.device, requires_grad=requires_grad)


def diag_last(x: torch.Tensor) -> torch.Tensor:
    """Make diagonal matrix from the last dimension of x, for any shape of x.

    Args:
        x: input tensor (..., D).
    Returns:
        diagonal tensor with the last dimension of x as its diagonal.
    """
    assert x.ndim > 0
    x_eye = eye(x.shape, dtype=x.dtype, device=x.device)
    return x_eye * x.unsqueeze(-1)
