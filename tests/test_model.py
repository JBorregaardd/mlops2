from mlopscookie.model import SimpleCNN
import torch

def test_model():
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)