import pytest
import torch
import torram

from typing import Tuple


@pytest.mark.parametrize("shape", ((1, 1, 1), (1, ), (7, 3, 5, 2)))
def test_diag_last_output_shape(shape: Tuple[int]):
    x = torch.rand(shape)
    y = torram.geometry.diag_last(x)
    assert y.shape == (*shape, shape[-1])


def test_diag_values():
    x = torch.rand((3, 3))
    y = torram.geometry.diag_last(x)
    for i in range(3):
        assert torch.allclose(y[i], torch.diag(x[i, :]))  # torch.diag(1D tensor) => diagonal matrix


@pytest.mark.parametrize("shape", ((1, ), (1, 1, 1), (7, 5, 3, 2)))
def test_eye(shape: Tuple[int]):
    x = torram.geometry.eye(shape)
    n = shape[-1]
    assert x.shape[:-1] == shape
    assert x.shape[-1] == n

    for i in range(n):
        for j in range(n):
            if i == j:
                expected = torch.ones_like(x[..., i, j])
            else:
                expected = torch.zeros_like(x[..., i, j])
            assert torch.allclose(x[..., i, j], expected)
