"""Small, independently testable PyTorch model for the XOR task."""

from torch import Tensor, nn


class XORNetwork(nn.Module):
    """A multilayer perceptron that classifies two-bit XOR inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)
