import pytest
import torch
import torram

from torch.utils.data import TensorDataset
from typing import Dict


class Model(torch.nn.Module):

    @staticmethod
    def wbatch(batch):
        x, y = batch
        return x + y

    @staticmethod
    def evaluate(batch, model_output, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        x, y = batch
        diff = (x + y) - model_output
        if reduce_mean:
            diff = torch.mean(diff)
        return {"z1": diff, "z2": diff ** 2}


@pytest.mark.parametrize("shape", ((4,), (5, 8), (1, 1)))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_evaluator(shape, batch_size):
    model = Model()
    dataset = TensorDataset(torch.rand(shape), torch.rand(shape))
    results = torram.evaluate_dataset(model, dataset, batch_size=batch_size)
    assert list(results.keys()) == ["z1", "z2"]
    assert results['z1'].shape == shape
    assert results['z2'].shape == shape
