import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.utils.checkpoint import checkpoint

from commons import init_weights
from residuals import LRELU_SLOPE, ResBlock


class SineGen(nn.Module):
    """
    Definition of sine generator

    Generates sine waveforms with optional harmonics and additive noise.
    Can be used to create harmonic noise source for neural vocoders.

    Args:
        samp_rate (int): Sampling rate in Hz.
        harmonic_num (int): Number of harmonic overtones (default 0).
        sine_amp (float): Amplitude of sine-waveform (default 0.1).
        noise_std (float): Standard deviation of Gaussian noise (default 0.003).
        voiced_threshold (float): F0 threshold for voiced/unvoiced classification (default 0).
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.merge = nn.Sequential(
            nn.Linear(self.dim, 1, bias=False),
            nn.Tanh(),
        )

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The integer part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        tmp_over_one = torch.cumsum(rad_values, 1) % 1
        tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
        cumsum_shift = torch.zeros_like(rad_values)
        cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

        sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)

        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            uv = self._f02uv(f0)

            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise
        # correct DC offset
        sine_waves = sine_waves - sine_waves.mean(dim=1, keepdim=True)
        # merge with grad
        return self.merge(sine_waves)


class HiFiGANNSFGenerator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        upsample_rates: list,
        upsample_initial_channel: int,
        upsample_kernel_sizes: list,
        gin_channels: int,
        sr: int,
        checkpointing: bool = False,
        num_streams: int = 4,
    ):
        super(HiFiGANNSFGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.checkpointing = checkpointing
        self.num_streams = num_streams
        self.f0_upsamp = torch.nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SineGen(sample_rate=sr)

        self.conv_pre = torch.nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = torch.nn.ModuleList()
        self.noise_convs = torch.nn.ModuleList()

        channels = [
            upsample_initial_channel // (2 ** (i + 1))
            if i < self.num_upsamples - 1
            else self.num_streams
            for i in range(len(upsample_rates))
        ]
        stride_f0s = [
            math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1
            for i in range(len(upsample_rates))
        ]

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            in_channels=upsample_initial_channel // (2**i),
                            out_channels=channels[i] * (
                                u**2
                            ),
                            kernel_size=k,
                            stride=1,
                            padding=(k - 1) // 2
                        )
                    ),
                    nn.PixelShuffle(upscale_factor=u),
                )
            )

            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            padding = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(
                nn.Conv1d(
                    1,
                    channels[i],
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                )
            )

        self.resblocks = torch.nn.ModuleList(
            [
                ResBlock(channels[i], k, d)
                for i in range(len(self.ups))
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ]
        )

        self.stream_combiner = torch.nn.Conv1d(
            in_channels=num_streams,
            out_channels=1,
            kernel_size=63,
            stride=1,
            padding=31,
            bias=False,
        )

        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)
        self.lrelu_slope = LRELU_SLOPE

    def forward(
        self, x: torch.Tensor, f0: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        f0 = F.interpolate(
            f0.unsqueeze(1), size=x.shape[-1] * self.upp, mode="linear"
        )
        har_source = self.m_source(f0.transpose(1, 2)).transpose(1, 2)
        x = self.conv_pre(x)

        if g is not None:
            x += self.cond(g)

        for i, (ups, noise_convs) in enumerate(zip(self.ups, self.noise_convs)):
            # in-place call
            x = torch.nn.functional.leaky_relu_(x, self.lrelu_slope)

            # Apply upsampling layer
            if self.training and self.checkpointing:
                x = checkpoint(ups, x, use_reentrant=False)
            else:
                x = ups(x)

            # Add noise excitation
            x += noise_convs(har_source)

            # Apply residual blocks
            def resblock_forward(x, blocks):
                return sum(block(x) for block in blocks) / len(blocks)

            blocks = self.resblocks[i * self.num_kernels : (i + 1) * self.num_kernels]

            # Checkpoint or regular computation for ResBlocks
            if self.training and self.checkpointing:
                x = checkpoint(resblock_forward, x, blocks, use_reentrant=False)
            else:
                x = resblock_forward(x, blocks)

        x = torch.nn.functional.leaky_relu_(x)
        x = torch.tanh_(self.stream_combiner(x))
        return x

    def remove_weight_norm(self):
        for l in self.ups:
            if hasattr(l, 'weight_norm'):
                remove_weight_norm(l)
            for layer in l:
                if hasattr(layer, 'weight_norm'):
                    remove_weight_norm(layer)
        for l in self.resblocks:
            l.remove_weight_norm()


    def __prepare_scriptable__(self):
        for l in self.ups:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        for l in self.resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"
                ):
                    remove_weight_norm(l)
        return self