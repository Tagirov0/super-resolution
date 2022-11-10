import torch
import numpy as np
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            residual_blocks=16
    ):
        super().__init__()

        self.initial = nn.Sequential(nn.Conv2d(in_channels, 64, 9, 1, 4), nn.PReLU(64))
        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(residual_blocks)])
        self.conv = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * 4, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(64),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(upscale_factor=2),
            nn.PReLU(64)
        )
        self.final = nn.Sequential(nn.Conv2d(64, 3, 9, 1, 4), nn.Tanh())

    def forward(self, x):
        initial = self.initial(x)
        out2 = self.residuals(initial)
        out3 = initial + self.conv(out2)
        out4 = self.upsample(out3)
        return self.final(out4)

class Discriminator(nn.Module):
    def __init__(
            self,
            h_size,
            w_size,
            in_channels=64,
            channels=[64, 64, 128, 128, 256, 256, 512, 512]
    ):
        super().__init__()

        size = int(np.ceil(h_size / 2 ** len(set(channels)))
                   * np.ceil(w_size / 2 ** len(set(channels))))

        self.initial = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels[0], channels[1], 3, 2, 1),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(0.2, inplace=True)
        )

        blocks = []
        for idx, channel in enumerate(channels[2:]):
            blocks.extend([
                nn.Conv2d(in_channels, channel, kernel_size=3, stride=1 + idx % 2, padding=1),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = channel

        self.conv_blocks = nn.Sequential(*blocks)

        self.final = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * size, out_features=1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=1)
        )

    def forward(self, x):
        out1 = self.initial(x)
        out2 = self.conv_blocks(out1)
        return self.final(out2)