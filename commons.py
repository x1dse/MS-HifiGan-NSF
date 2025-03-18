import torch
from typing import Optional


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize the weights of a module.

    Args:
        m: The module to initialize.
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """
    Calculate the padding needed for a convolution.

    Args:
        kernel_size: The size of the kernel.
        dilation: The dilation of the convolution.
    """
    return int((kernel_size * dilation - dilation) / 2)