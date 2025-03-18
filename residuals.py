import torch
from itertools import chain
from typing import Optional, Tuple
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from commons import get_padding, init_weights

LRELU_SLOPE = 0.1

def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


def apply_mask_(tensor: torch.Tensor, mask: Optional[torch.Tensor]):
    return tensor.mul_(mask) if mask else tensor


class ResBlock(torch.nn.Module):
    """
    A residual block module that applies a series of 1D convolutional layers with residual connections.
    """

    def __init__(
        self, channels: int, kernel_size: int = 3, dilations: Tuple[int] = (1, 3, 5)
    ):
        """
        Initializes the ResBlock.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel. Defaults to 3.
            dilations (Tuple[int]): Tuple of dilation rates for the convolution layers in the first set.
        """
        super().__init__()
        # Create convolutional layers with specified dilations and initialize weights
        self.convs1 = self._create_convs(channels, kernel_size, dilations)
        self.convs2 = self._create_convs(channels, kernel_size, [1] * len(dilations))

    @staticmethod
    def _create_convs(channels: int, kernel_size: int, dilations: Tuple[int]):
        """
        Creates a list of 1D convolutional layers with specified dilations.

        Args:
            channels (int): Number of input and output channels for the convolution layers.
            kernel_size (int): Size of the convolution kernel.
            dilations (Tuple[int]): Tuple of dilation rates for each convolution layer.
        """
        layers = torch.nn.ModuleList(
            [create_conv1d_layer(channels, kernel_size, d) for d in dilations]
        )
        layers.apply(init_weights)
        return layers

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            x_residual = x
            # new tensor
            x = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            # in-place call
            x = apply_mask_(x, x_mask)
            # in-place call
            x = torch.nn.functional.leaky_relu_(conv1(x), LRELU_SLOPE)
            # in-place call
            x = apply_mask_(x, x_mask)
            x = conv2(x)
            # in-place call
            x += x_residual
        # in-place call
        return apply_mask_(x, x_mask)

    def remove_weight_norm(self):
        for conv in chain(self.convs1, self.convs2):
            remove_weight_norm(conv)