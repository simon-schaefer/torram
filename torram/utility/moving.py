import logging
import torch


def move_batch(batch, device: torch.device):
    if hasattr(batch, "to"):
        return batch.to(device)
    elif isinstance(batch, tuple):
        return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
    elif isinstance(batch, dict):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    logging.warning(f"Moving batch data not implemented for batch type {type(batch)}")
    return batch
