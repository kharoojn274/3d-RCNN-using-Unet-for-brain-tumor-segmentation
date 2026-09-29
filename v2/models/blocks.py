from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct3D(nn.Module):
    """3D convolution followed by instance normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.norm = nn.InstanceNorm3d(
            out_channels,
            affine=True,
        )

        self.activation = (
            nn.LeakyReLU(0.01, inplace=True)
            if activation
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class RecurrentConv3D(nn.Module):
    """Recurrent convolutional feature refinement.

    The same convolutional transformation is applied repeatedly,
    allowing the block to refine spatial features over multiple steps.
    """

    def __init__(
        self,
        channels: int,
        steps: int = 2,
    ):
        super().__init__()

        if steps < 1:
            raise ValueError("steps must be >= 1")

        self.steps = steps

        self.conv = ConvNormAct3D(
            channels,
            channels,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x

        for _ in range(self.steps):
            output = self.conv(output)

        return output


class RCNNBlock3D(nn.Module):
    """3D recurrent convolutional residual block.

    Structure:
        input
          -> projection
          -> recurrent convolutions
          -> residual addition
          -> activation

    This provides the RCNN component used by the V2 encoder/decoder.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        recurrent_steps: int = 2,
    ):
        super().__init__()

        self.projection = (
            nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.InstanceNorm3d(
                    out_channels,
                    affine=True,
                ),
            )
            if in_channels != out_channels
            else nn.Identity()
        )

        self.input_activation = nn.LeakyReLU(
            0.01,
            inplace=True,
        )

        self.initial = ConvNormAct3D(
            in_channels,
            out_channels,
        )

        self.recurrent = RecurrentConv3D(
            out_channels,
            steps=recurrent_steps,
        )

        self.output_activation = nn.LeakyReLU(
            0.01,
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)

        output = self.initial(x)
        output = self.recurrent(output)

        output = output + residual

        return self.output_activation(output)


class Downsample3D(nn.Module):
    """Strided 3D downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                0.01,
                inplace=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock3D(nn.Module):
    """3D transposed-convolution upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
