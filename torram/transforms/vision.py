import torch
import torchvision.transforms.functional

__all__ = ['normalize_images']


def normalize_images(x: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    batch_size = x.shape[0]
    for i in range(batch_size):
        x[i] = torchvision.transforms.functional.normalize(x[i], mean=mean, std=std)
    return x
