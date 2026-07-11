import torch


def test_torch_cpu_tensor():
    x = torch.randn(3, 3)
    y = x @ x.T
    assert y.shape == (3, 3)


def test_torch_is_cpu_build():
    assert not torch.cuda.is_available()