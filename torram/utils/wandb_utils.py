import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb


def load_checkpoint_from_wandb(
    wandb_path: str,
    device: torch.device,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Load a model checkpoint from the local W&B run directory.

    @param wandb_path: Path to the W&B run, e.g., "username/project/run_id".
    @param device: Device to load the checkpoint onto (e.g., "cuda" or "cpu").
    @param step: Optional step number to load a specific checkpoint. If None, loads the latest checkpoint.
    @return: Loaded checkpoint dictionary.
    """
    logger = logging.getLogger(__name__)
    api = wandb.Api()
    run = api.run(wandb_path)

    # Parse the local model log path from the W&B run.
    dt = datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%SZ")
    dir_name = dt.strftime(f"run-%Y%m%d_%H%M%S-{run.id}")
    checkpoint_dir = Path(run.dir) / dir_name / "checkpoints"

    # Either use the provided step or use the largest step from the run.
    if step is None:
        step = max(int(f.stem.split("-")[-1]) for f in checkpoint_dir.glob("*.pt"))
    checkpoint_file = checkpoint_dir / f"checkpoint_{step}.pt"

    logger.info(f"Loading checkpoint from {checkpoint_file}")
    return torch.load(checkpoint_file, map_location=device)
