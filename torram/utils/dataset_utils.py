import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset


def train_test_split(dataset, test_ratio: float, seed: int = 42) -> Tuple[Subset, Subset]:
    total_size = len(dataset)
    indices = list(range(total_size))

    # Shuffle indices with seed and new random instance (to avoid
    # affecting / being dependent on global random state).
    rng = random.Random(seed)
    rng.shuffle(indices)

    # Split indices into train and test sets.
    split = int(total_size * (1 - test_ratio))
    train_indices = indices[:split]
    test_indices = indices[split:]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def get_batch_from_dataset(
    dataset,
    batch_size: int,
    indices: Optional[List[int]] = None,
    collate_fn=None,
):
    """Get a batch from a dataset with the given batch size.

    @param dataset: The dataset to sample from.
    @param batch_size: The number of samples in the batch.
    @param indices: Optional list of indices to sample. If None, random indices will be chosen.
    @param collate_fn: Optional function to collate the batch.
    """
    if indices is None:
        indices = list(range(len(dataset)))
        indices = random.sample(indices, batch_size)
    batch = [dataset[i] for i in indices]

    # Collate batch (assuming each item is a dict).
    if collate_fn:
        return collate_fn(batch)

    # Default collation, stack values by key.
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return {key: torch.stack(collated_batch[key], dim=0) for key in collated_batch}


def chunk_and_save(
    data: Dict[str, torch.Tensor],
    output_dir: Path,
    seq_len: int,
    suffix: str = ".pt",
    include_incomplete: bool = False,
) -> List[Path]:
    """Chunk data into sequences of fixed length and save to disk.

    @param data: Dictionary of data arrays to chunk. All arrays must have the same length.
    @param output_dir: Directory to save the chunked data.
    @param seq_len: Length of each chunked sequence.
    @param include_incomplete: Whether to include the last chunk if it's shorter than seq_len.
    """
    logger = logging.getLogger(__name__)
    if len(data) == 0:
        logger.warning("No data provided to chunk_and_save.")
        return []

    num_frames = len(next(iter(data.values())))
    assert all(len(value) == num_frames for value in data.values())

    output_files = []
    for start_idx in range(0, num_frames, seq_len):
        end_idx = min(start_idx + seq_len, num_frames)
        if end_idx - start_idx < seq_len and not include_incomplete:
            continue

        output_file = output_dir / f"{start_idx:06d}_{end_idx:06d}{suffix}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value[start_idx:end_idx].clone() for key, value in data.items()},
            output_file,
        )
        output_files.append(output_file)

    return output_files
