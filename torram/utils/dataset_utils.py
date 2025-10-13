import random
from typing import List, Optional, Tuple

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
