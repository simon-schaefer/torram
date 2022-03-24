import abc
import dataclasses
import torch

from typing import Callable


class TensorTuple(abc.ABC):
    """Extend dataclasses with tensor fields with basic tensor operations.

    >>> @dataclasses.dataclass
    ... class ExampleClass(TensorTuple):
    ...     a: torch.Tensor
    ...     b: torch.Tensor
    ...     c: str

    >>> obj = ExampleClass(torch.zeros(3), torch.ones(3), "abc")
    >>> obj[:2]
    ExampleClass(a=tensor([0., 0.]), b=tensor([1., 1.]), c='abc')
    >>> obj.to('cuda:0')
    ExampleClass(a=tensor([0., 0., 0.], device='cuda:0'), b=tensor([1., 1., 1.], device='cuda:0'), c='abc')
    """

    def __apply_to_tensor_fields(self, func: Callable[[torch.Tensor], torch.Tensor]):
        init_dict = {}
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                init_dict[field] = func(value)
            else:
                init_dict[field] = value
        return self.__class__(**init_dict)

    @property
    def device(self):
        devices = [getattr(self, field).device for field in self.__dataclass_fields__
                   if isinstance(getattr(self, field), torch.Tensor)]

        all_devices_identical = all([devices[0] == device for device in devices])
        if not all_devices_identical:
            raise RuntimeError(f"Tensors in {self.__class__.__name__} are on different devices")
        return devices[0]

    def to(self, device):
        return self.__apply_to_tensor_fields(lambda x: x.to(device))

    def detach(self):
        return self.__apply_to_tensor_fields(lambda x: x.detach())

    def clone(self):
        return self.__apply_to_tensor_fields(lambda x: x.clone())

    def __getitem__(self, i):
        return self.__apply_to_tensor_fields(lambda x: x[i])
