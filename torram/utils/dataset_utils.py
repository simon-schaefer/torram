import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sized, Tuple, Union, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

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

    def get_train_test_split(
        self,
    ) -> Tuple[
        torch.utils.data.Dataset,
        Union[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]],
    ]:
        """Return train and test splits of the dataset."""
        raise NotImplementedError

    def collate_fn(self, batch: List[Any]) -> Any:
        """Collate function to combine a list of samples into a batch."""
        raise NotImplementedError

    def augment_fn(self, batch: Any, iteration: int) -> Any:
        """Augmentation function to augment a batch based on the training iteration."""
        raise NotImplementedError


class CustomSchemeDataLoader(DataLoader):
    """DataLoader that loads the functions defined in the dataset schema."""

    def __init__(self, dataset: Dataset, **kwargs):
        assert "collate_fn" not in kwargs, "collate_fn should not be passed explicitly."
        collate_fn = get_collate_fn(dataset)
        super().__init__(dataset=dataset, collate_fn=collate_fn, **kwargs)

        self.augment_fn = get_augment_fn(dataset)
        self.iteration = 0

    def __iter__(self):
        for batch in super().__iter__():
            if self.augment_fn:
                batch = self.augment_fn(batch, self.iteration)
            yield batch

    def set_iteration(self, iteration: int) -> None:
        self.iteration = iteration


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
    augment_index: int = 0,
    seed: Optional[int] = None,
):
    """Get a batch from a dataset with the given batch size.

    @param dataset: The dataset to sample from.
    @param batch_size: The number of samples in the batch.
    @param indices: Optional list of indices to sample. If None, random indices will be chosen.
    @param augment_index: The iteration index for augmentation (if applicable).
    @param seed: Optional random seed for sampling indices.
    """
    if indices is None:
        if seed is not None:
            random.seed(seed)
        indices = list(range(len(dataset)))
        indices = random.sample(indices, batch_size)
    batch = [dataset[i] for i in indices]

    # Collate batch (assuming each item is a dict).
    collate_fn = get_collate_fn(dataset)
    if collate_fn:
        batch = collate_fn(batch)

    # Default collation, stack values by key.
    else:
        collated_batch = {}
        for key in batch[0].keys():
            collated_batch[key] = [item[key] for item in batch]
        batch = {key: torch.stack(collated_batch[key], dim=0) for key in collated_batch}

    # Augment batch if augment function is defined.
    augment_fn = get_augment_fn(dataset)
    if augment_fn:
        batch = augment_fn(batch, iteration=augment_index)

    return batch


def get_test_batch_from_dataset(
    dataset,
    batch_size: int,
    indices: Optional[List[int]] = None,
    seed: Optional[int] = None,
):
    """Get a test batch from a dataset with the given batch size. This assumes the dataset
    has a `get_train_test_split` method that returns train and test splits. If multiple
    test sets are returned, the dataset is concatenated and indexed (either randomly or by provided indices).

    @param dataset: The dataset to sample from.
    @param batch_size: The number of samples in the batch.
    @param indices: Optional list of indices to sample. If None, random indices will be chosen.
    @param seed: Optional random seed for sampling indices.
    """
    _, test_datasets = get_train_test_split_w_ratio(dataset, test_ratio=0.1)
    ds = ConcatDataset(list(test_datasets.values()))
    return get_batch_from_dataset(ds, batch_size=batch_size, indices=indices, seed=seed)


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
) -> Tuple[torch.utils.data.Dataset, Dict[str, torch.utils.data.Dataset]]:
    """Split a ConcatDataset into train and test sets.

    If any of the constituent datasets have a predefined train/test split method,
    those splits will be used. Datasets without predefined splits will be included
    in the training set only.

    @param dataset: The ConcatDataset to split.
    @param test_ratio: The ratio of the dataset to use for testing.
    @param seed: Random seed for splitting.
    """
    logger = logging.getLogger(__name__)
    has_predefined_splits = any(hasattr(ds, "get_train_test_split") for ds in dataset.datasets)

    # If no datasets have predefined splits, do a random split on the concatenated dataset.
    if not has_predefined_splits:
        train_ds, test_ds = train_test_split(dataset, test_ratio, seed)
        return train_ds, {"": test_ds} if len(test_ds) > 0 else {}

    # If some datasets have predefined splits, use them. All datasets with no predefined splits
    # will be included in the training set only.
    train_datasets = []
    test_datasets = {}
    for ds in dataset.datasets:
        if hasattr(ds, "get_train_test_split"):
            train_ds, test_ds = ds.get_train_test_split()
            train_datasets.append(train_ds)
            if not isinstance(test_ds, dict):
                test_ds = {"": test_ds}

            for k in test_ds:
                assert k not in test_datasets, f"Duplicate test dataset key {k} found."

            test_datasets.update({k: v for k, v in test_ds.items() if len(v) > 0})

        else:
            logger.debug(f"Dataset {ds} wo/ predefined split, including as training set only.")
            train_datasets.append(ds)

    train_ds = torch.utils.data.ConcatDataset(train_datasets)
    return train_ds, test_datasets


def get_train_test_split_w_config(
    dataset: torch.utils.data.Dataset,
    config: DataConfig,
) -> Tuple[torch.utils.data.Dataset, Mapping[str, torch.utils.data.Dataset]]:
    """Get train and test splits of a dataset based on the provided configuration.

    @param dataset: The dataset to split.
    @param config: DataConfig containing the test split ratio.
    """
    return get_train_test_split_w_ratio(dataset=dataset, test_ratio=config.test_split)


def get_train_test_split_w_ratio(
    dataset: torch.utils.data.Dataset,
    test_ratio: float,
) -> Tuple[torch.utils.data.Dataset, Mapping[str, torch.utils.data.Dataset]]:
    """Get train and test splits of a dataset based test set ratio (if split undefined).

    @param dataset: The dataset to split.
    @param test_ratio: The ratio of the dataset to use for testing.
    """
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        dataset_train, datasets_test = train_test_split_concat_dataset(
            dataset=dataset,
            test_ratio=test_ratio,
        )
    elif hasattr(dataset, "get_train_test_split"):
        dataset_train, datasets_test = getattr(dataset, "get_train_test_split")()
        if not isinstance(datasets_test, dict):
            datasets_test = {"": datasets_test}
    else:
        dataset_train, dataset_test_ = train_test_split(dataset, test_ratio=test_ratio)
        dataset_test_ = cast(torch.utils.data.Dataset, dataset_test_)
        datasets_test = {"": dataset_test_}

    datasets_test = {
        k: v
        for k, v in datasets_test.items()
        if (isinstance(v, Sized) and isinstance(v, torch.utils.data.Dataset) and len(v) > 0)
    }
    return dataset_train, datasets_test


def _get_fn_from_dataset(dataset, fn_name: str):
    """Get a function from a dataset, if it has one.

    @param dataset: The dataset to get the function from.
    @param fn_name: The name of the function to get.
    """
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        # Check if all constituent datasets have the same function.
        fns = [_get_fn_from_dataset(ds, fn_name) for ds in dataset.datasets]
        assert len(fns) > 0
        if (
            all(fns[0] is cf for cf in fns)
            or all((cf is None for cf in fns))
            or all((cf.__code__ == fns[0].__code__ for cf in fns))
        ):
            return fns.pop()
        else:
            raise ValueError(f"All datasets in ConcatDataset must have the same {fn_name}.")

    if isinstance(dataset, torch.utils.data.Subset):
        return _get_fn_from_dataset(dataset.dataset, fn_name)

    return getattr(dataset, fn_name, None)


def get_collate_fn(dataset: torch.utils.data.Dataset):
    """Get the collate function for a dataset, if it has one.

    @param dataset: The dataset to get the collate function from.
    """
    return _get_fn_from_dataset(dataset, "collate_fn")


def get_augment_fn(dataset: torch.utils.data.Dataset):
    """Get the augment function for a dataset, if it has one.

    @param dataset: The dataset to get the augment function from.
    """
    return _get_fn_from_dataset(dataset, "augment_fn")
