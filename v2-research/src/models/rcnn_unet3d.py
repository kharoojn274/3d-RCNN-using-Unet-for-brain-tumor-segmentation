"""Modular 3D RCNN-U-Net foundation for V2.

This is the research architecture scaffold. Exact channel counts, recurrence,
losses, and task-specific heads will be tuned after inspecting the 2026 task
labels and training data.
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RecurrentConvBlock3D(nn.Module):
    """Lightweight recurrent convolutional refinement block."""
    def __init__(self, channels, steps=2):
        super().__init__()
        self.steps = steps
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        state = x
        for _ in range(self.steps):
            state = self.act(self.norm(self.conv(state + x)))
        return state


class RCNNUNet3D(nn.Module):
    """3D recurrent-convolutional U-Net segmentation model."""
    def __init__(self, in_channels=4, num_classes=4, base_channels=16,
                 recurrent_steps=2):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock3D(in_channels, c)
        self.rec1 = RecurrentConvBlock3D(c, recurrent_steps)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock3D(c, c * 2)
        self.rec2 = RecurrentConvBlock3D(c * 2, recurrent_steps)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.rec3 = RecurrentConvBlock3D(c * 4, recurrent_steps)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(c * 4, c * 8)
        self.rec4 = RecurrentConvBlock3D(c * 8, recurrent_steps)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(c * 4, c * 2)

        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock3D(c * 2, c)

        self.head = nn.Conv3d(c, num_classes, 1)

    @staticmethod
    def _match(x, ref):
        # Center-crop/pad-free alignment for odd input dimensions.
        d = min(x.shape[2], ref.shape[2])
        h = min(x.shape[3], ref.shape[3])
        w = min(x.shape[4], ref.shape[4])
        x = x[:, :, :d, :h, :w]
        ref = ref[:, :, :d, :h, :w]
        return x, ref

    def forward(self, x):
        e1 = self.rec1(self.enc1(x))
        e2 = self.rec2(self.enc2(self.pool1(e1)))
        e3 = self.rec3(self.enc3(self.pool2(e2)))
        b = self.rec4(self.bottleneck(self.pool3(e3)))

        d3 = self.up3(b)
        d3, e3a = self._match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))

        d2 = self.up2(d3)
        d2, e2a = self._match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))

        d1 = self.up1(d2)
        d1, e1a = self._match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))

        return self.head(d1)
