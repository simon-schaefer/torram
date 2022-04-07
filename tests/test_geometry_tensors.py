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
