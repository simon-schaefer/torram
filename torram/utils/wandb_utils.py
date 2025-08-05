import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch
import wandb
from wandb.apis.public.files import File


def load_checkpoint_from_wandb(
    wandb_path: str,
    device: torch.device,
    step: Optional[int] = None,
    download_dir: Path = Path("/tmp/wandb_checkpoints"),
    replace: bool = False,
) -> Dict[str, Any]:
    """Load a model checkpoint from the local W&B run directory.

    @param wandb_path: Path to the W&B run, e.g., "username/project/run_id".
    @param device: Device to load the checkpoint onto (e.g., "cuda" or "cpu").
    @param step: Optional step number to load a specific checkpoint. If None, loads the latest checkpoint.
    @param download_dir: Directory to download the checkpoint file to.
    @param replace: Whether to replace the existing file if it already exists.
    @return: Loaded checkpoint dictionary.
    """
    logger = logging.getLogger(__name__)
    api = wandb.Api()
    run = api.run(wandb_path)

    # Either use the provided step or use the largest step from the run.
    if step is None:
        files = run.files()
        checkpoint_files = [f for f in files if f.name.endswith(".pt")]
        checkpoint_files.sort(key=lambda f: f.updated_at, reverse=True)
        if not checkpoint_files:
            raise ValueError("No checkpoint files found in the W&B run.")

        latest_checkpoint = checkpoint_files[0]

    else:
        latest_checkpoint = run.file(f"files/checkpoint_{step}.pt")
        if not latest_checkpoint.exists():
            raise ValueError(f"Checkpoint for step {step} does not exist in the W&B run.")
    latest_checkpoint = cast(File, latest_checkpoint)
    logger.info(f"Loading checkpoint from W&B run: {latest_checkpoint.name}")

    # Download the checkpoint file to the specified directory.
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
    download_path = download_dir.as_posix()

    cache_file = latest_checkpoint.download(root=download_path, replace=replace, exist_ok=True)
    cache_file_name = cache_file.name
    logger.debug(f"Checkpoint downloaded to: {cache_file_name}")

    # Load the checkpoint file.
    return torch.load(cache_file_name, map_location=device)
