import collections
import torch
import torch.utils.data

from torram.utility.moving import move_batch
from typing import Any, Dict, Optional, Protocol


class EvaluatorFunction(Protocol):

    def __call__(self, batch: Any, model_output: Any, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        ...


class EvaluatorFactory:

    def __init__(self, eval_function: EvaluatorFunction, device: torch.device):
        self.eval_function = eval_function
        self.device = device

    def __call__(self, batch: Any, model_output: Any, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        return self.eval_function(batch, model_output, reduce_mean=reduce_mean)

    def wmodel(self, model, batch: Any, reduce_mean: bool = True) -> Dict[str, torch.Tensor]:
        model_output = model(batch)
        return self.eval_function(batch, model_output, reduce_mean=reduce_mean)

    def wloader(self, model, data_loader: torch.utils.data.DataLoader, until: Optional[int] = None,
                dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        if until is not None and until <= 0:
            raise ValueError(f"Invalid value for until, expected integer > 0, got {until}")

        metric_cache = collections.defaultdict(lambda: torch.zeros(0, dtype=dtype, device=self.device))
        for j, batch in enumerate(data_loader):
            if until is not None and j > until:
                break
            batch = move_batch(batch, device=self.device)
            metrics_dict = self.wmodel(model, batch, reduce_mean=False)
            metric_cache = {key: torch.cat([metric_cache[key], value]) for key, value in metrics_dict.items()}
        return {key: torch.mean(value) for key, value in metric_cache.items()}
