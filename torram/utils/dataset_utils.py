import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data import Subset

from torram.utils.ops import collect_nested_leaf_lengths, slice_stacked


@dataclass
class DataConfig:
    num_workers: int
    batch_size: int
    test_split: float


class DatasetSchema(Protocol):

    def __init__(self, config: Any):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class ExtendedDatasetSchema(DatasetSchema, Protocol):

    def get_train_test_split(self) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Return train and test splits of the dataset."""
        raise NotImplementedError


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
    data: Dict[str, torch.Tensor | np.ndarray | Dict],
    output_dir: Path,
    seq_len: int,
    stride: Optional[int] = None,
    suffix: str = ".pt",
    include_incomplete: bool = False,
) -> List[Path]:
    """Chunk data into sequences of fixed length and save to disk.

    @param data: Dictionary of data arrays to chunk. All leafs must have the same length.
    @param output_dir: Directory to save the chunked data.
    @param seq_len: Length of each chunked sequence.
    @param stride: Stride between chunks. If None, defaults to seq_len (non-overlapping).
    @param include_incomplete: Whether to include the last chunk if it's shorter than seq_len.
    """
    logger = logging.getLogger(__name__)
    if len(data) == 0:
        logger.warning("No data provided to chunk_and_save.")
        return []
    stride = seq_len if stride is None else stride

    leaf_lengths = collect_nested_leaf_lengths(data)
    if len(set(leaf_lengths)) != 1:
        raise ValueError(f"All data leafs must have the same length, got lengths: {leaf_lengths}")
    num_frames = leaf_lengths[0]

    output_files = []
    for start_idx in range(0, num_frames, stride):
        end_idx = min(start_idx + seq_len, num_frames)
        if end_idx - start_idx < seq_len and not include_incomplete:
            continue
        data_sliced = slice_stacked(data, start_idx, end_idx)

        output_file = output_dir / f"{start_idx:06d}_{end_idx:06d}{suffix}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data_sliced, output_file)
        output_files.append(output_file)

    return output_files


def train_test_split_concat_dataset(
    dataset: torch.utils.data.ConcatDataset,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Split a ConcatDataset into train and test sets.

    If any of the constituent datasets have a predefined train/test split method,
    those splits will be used. Datasets without predefined splits will be included
    in the training set only.

    @param dataset: The ConcatDataset to split.
    @param test_ratio: The ratio of the dataset to use for testing.
    @param seed: Random seed for splitting.
    """
    has_predefined_splits = any(hasattr(ds, "get_train_test_split") for ds in dataset.datasets)

    # If no datasets have predefined splits, do a random split on the concatenated dataset.
    if not has_predefined_splits:
        return train_test_split(dataset, test_ratio, seed)

    # If some datasets have predefined splits, use them. All datasets with no predefined splits
    # will be included in the training set only.
    train_datasets = []
    test_datasets = []
    for ds in dataset.datasets:
        if hasattr(ds, "get_train_test_split"):
            train_ds, test_ds = ds.get_train_test_split()
            train_datasets.append(train_ds)
            test_datasets.append(test_ds)
        else:
            train_datasets.append(ds)

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    test_ds = torch.utils.data.ConcatDataset(test_datasets)
    return train_ds, test_ds


def get_train_test_split_w_config(
    dataset: torch.utils.data.Dataset,
    config: DataConfig,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Get train and test splits of a dataset based on the provided configuration.

    @param dataset: The dataset to split.
    @param config: DataConfig containing the test split ratio.
    """
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        dataset_train, dataset_test = train_test_split_concat_dataset(
            dataset=dataset,
            test_ratio=config.test_split,
        )
    elif hasattr(dataset, "get_train_test_split"):
        dataset_train, dataset_test = dataset.get_train_test_split()
    else:
        dataset_train, dataset_test = train_test_split(dataset, test_ratio=config.test_split)
    return dataset_train, dataset_test
