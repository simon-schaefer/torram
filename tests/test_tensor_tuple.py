import dataclasses
import torram
import torch


@dataclasses.dataclass(frozen=True)
class ExampleClass(torram.utility.TensorTuple):
    a: torch.Tensor
    b: torch.Tensor
    c: str


def test_to():
    test_object = ExampleClass(torch.zeros(5), torch.ones(5), "bread")
    if torch.cuda.is_available():
        test_object2 = test_object.to('cuda:0')
        assert test_object2.a.device == torch.device('cuda:0')


def test_index():
    test_object = ExampleClass(torch.zeros(5), torch.ones(5), "bread")
    test_object2: ExampleClass = test_object[:2]
    assert test_object2.a.shape == torch.Size((2, ))
