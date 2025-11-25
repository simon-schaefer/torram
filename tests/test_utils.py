from typing import Tuple

import pytest
import torch

from torram.utils.ops import diag_last, eye, eye_like


def is_eye_matrix(x: torch.Tensor) -> bool:
    n = x.shape[-1]
    if x.shape[-1] != x.shape[-2]:
        return False
    for i in range(n):
        for j in range(n):
            if i == j:
                expected = torch.ones_like(x[..., i, j])
            else:
                expected = torch.zeros_like(x[..., i, j])
            if not torch.allclose(x[..., i, j], expected):
                return False
    return True


@pytest.mark.parametrize("shape", ((1, 1, 1), (1,), (7, 3, 5, 2)))
def test_diag_last_output_shape(shape: Tuple[int]):
    x = torch.rand(shape)
    y = diag_last(x)
    assert y.shape == (*shape, shape[-1])


def test_diag_last_values():
    x = torch.rand((3, 3))
    y = diag_last(x)
    for i in range(3):
        assert torch.allclose(y[i], torch.diag(x[i, :]))  # torch.diag(1D tensor) => diagonal matrix


def test_diag_last_grad_safe():
    a = torch.rand((3, 2, 4), requires_grad=True)
    out = diag_last(a)
    grad = torch.autograd.grad(out.sum(), a)[0]
    assert not torch.any(torch.isnan(grad))


@pytest.mark.parametrize("shape", ((1,), (1, 1, 1), (7, 5, 3, 2)))
def test_eye(shape: Tuple[int]):
    x = eye(shape)
    assert x.shape[:-1] == shape
    assert is_eye_matrix(x)


@pytest.mark.parametrize("shape", ((4, 1, 2, 2), (4, 4), (8, 6, 6)))
def test_eye_like(shape: Tuple[int]):
    x = torch.rand(shape)
    y = eye_like(x)
    assert x.shape == y.shape
    assert is_eye_matrix(y)
