import torch

__all__ = ['diag_last']


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
