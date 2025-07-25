import random

from torch.utils.data import Subset


def train_test_split(dataset, test_ratio: float, seed: int = 42):
    total_size = len(dataset)
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)

    split = int(total_size * (1 - test_ratio))
    train_indices = indices[:split]
    test_indices = indices[split:]

    return Subset(dataset, train_indices), Subset(dataset, test_indices)
