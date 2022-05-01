import torch.utils.data
import torram

from typing import Any, Protocol


class ModelProtocol(Protocol):  # pragma: no cover

    def wbatch(self, batch) -> Any:
        ...

    def visualize(self, batch, model_output, logger, global_step: int, prefix: str = ""):
        ...


@torch.no_grad()
def visualize_dataset(model: ModelProtocol, dataset: torch.utils.data.Dataset, output_path: str,
                      device: torch.device('cpu'), stop_after: int = -1):
    """Visualize model outputs for dataset.

    Args:
        model: prediction model (batch prediction and visualization function).
        dataset: dataset to visualize un-shuffled.
        output_path: path to output directory of visualizations.
        device: device used for processing.
        stop_after: stops after n samples, -1 = do not stop.
    """
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=1)
    logger = torram.utility.logger.ImageLogger(output_path)
    for k, batch in enumerate(data_loader):
        if stop_after != -1 and k > stop_after:
            break
        batch = torram.utility.moving.move_batch(batch, device=device)
        model_output = model.wbatch(batch)
        model.visualize(batch, model_output, logger=logger, global_step=k)
