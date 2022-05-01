import torch
import torch.utils.data
import tqdm

from typing import Any, Dict, Protocol

import torram.utility.moving


class ModelProtocol(Protocol):  # pragma: no cover

    def wbatch(self, batch) -> Any:
        ...

    def eval(self):
        ...

    def evaluate(self, batch, model_output, reduce_mean: bool) -> Dict[str, torch.Tensor]:
        ...


@torch.no_grad()
def evaluate_dataset(model: ModelProtocol, dataset: torch.utils.data.Dataset, device: torch.device = torch.device('cpu'),
                     batch_size: int = 1, stop_after: int = -1):
    """Visualize model outputs for dataset.

    Args:
        model: prediction model (batch prediction and visualization function).
        dataset: evaluation dataset.
        device: device used for processing.
        batch_size: processing batch size.
        stop_after: stops after n samples, -1 = do not stop.
    """
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    model.eval()
    results = []
    for k, batch in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        if stop_after != -1 and k > stop_after:
            break
        batch = torram.utility.moving.move_batch(batch, device=device)
        model_output = model.wbatch(batch)
        results_k = model.evaluate(batch, model_output, reduce_mean=False)
        results.append(results_k)
    return {key: torch.cat([results_k[key] for results_k in results], dim=0) for key in results[0].keys()}
