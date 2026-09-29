from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import (
    RCNNBlock3D,
    Downsample3D,
    UpBlock3D,
)


class DecoderBlock3D(nn.Module):
    """Upsampling + skip fusion + RCNN refinement."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        recurrent_steps: int = 2,
    ):
        super().__init__()

        self.up = UpBlock3D(
            in_channels,
            out_channels,
        )

        self.refine = RCNNBlock3D(
            out_channels + skip_channels,
            out_channels,
            recurrent_steps=recurrent_steps,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        x = self.up(x)

        # Handle possible one-voxel differences caused by odd dimensions.
        if x.shape[2:] != skip.shape[2:]:
            diff = [
                skip.size(i + 2) - x.size(i + 2)
                for i in range(3)
            ]

            padding = []
            for d in reversed(diff):
                before = d // 2
                after = d - before
                padding.extend([before, after])

            x = nn.functional.pad(x, padding)

        x = torch.cat([x, skip], dim=1)

        return self.refine(x)


class RCNNUNet3D(nn.Module):
    """3D RCNN-U-Net V2 for multimodal brain tumor segmentation.

    Input:
        [B, 4, D, H, W]

    Output:
        [B, num_classes, D, H, W]

    The network uses recurrent convolutional residual blocks
    throughout a 3D U-Net encoder/decoder.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 16,
        recurrent_steps: int = 2,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # Encoder
        self.enc1 = RCNNBlock3D(
            in_channels,
            c1,
            recurrent_steps,
        )
        self.down1 = Downsample3D(c1, c2)

        self.enc2 = RCNNBlock3D(
            c2,
            c2,
            recurrent_steps,
        )
        self.down2 = Downsample3D(c2, c3)

        self.enc3 = RCNNBlock3D(
            c3,
            c3,
            recurrent_steps,
        )
        self.down3 = Downsample3D(c3, c4)

        self.enc4 = RCNNBlock3D(
            c4,
            c4,
            recurrent_steps,
        )
        self.down4 = Downsample3D(c4, c5)

        # Bottleneck
        self.bottleneck = RCNNBlock3D(
            c5,
            c5,
            recurrent_steps,
        )

        # Decoder
        self.dec4 = DecoderBlock3D(
            c5,
            c4,
            c4,
            recurrent_steps,
        )

        self.dec3 = DecoderBlock3D(
            c4,
            c3,
            c3,
            recurrent_steps,
        )

        self.dec2 = DecoderBlock3D(
            c3,
            c2,
            c2,
            recurrent_steps,
        )

        self.dec1 = DecoderBlock3D(
            c2,
            c1,
            c1,
            recurrent_steps,
        )

        self.segmentation_head = nn.Conv3d(
            c1,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        # Bottleneck
        b = self.bottleneck(self.down4(e4))

        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.segmentation_head(d1)


def build_rcnn_unet3d(
    in_channels: int = 4,
    num_classes: int = 4,
    base_channels: int = 16,
    recurrent_steps: int = 2,
) -> RCNNUNet3D:
    """Factory function for the V2 model."""
    return RCNNUNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        recurrent_steps=recurrent_steps,
    )


if __name__ == "__main__":
    # Lightweight architecture sanity check.
    model = build_rcnn_unet3d(
        in_channels=4,
        num_classes=4,
        base_channels=8,
    )

    x = torch.randn(
        1,
        4,
        32,
        32,
        32,
    )

    with torch.no_grad():
        y = model(x)

    print("Input :", tuple(x.shape))
    print("Output:", tuple(y.shape))
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )
