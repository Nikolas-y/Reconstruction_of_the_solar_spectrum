import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, use_se=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, padding)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.se(x)
        return self.drop(x)

class SpectralUNet(nn.Module):
    def __init__(self, input_len=9, output_len=1904, base_channels=32):
        super().__init__()
        self.output_len = output_len
        self.expander = nn.Sequential(
            nn.Linear(input_len, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, base_channels * 16),
            nn.GELU()
        )
        # Encoder
        self.enc1 = ConvBlock(base_channels, base_channels*2, stride=2)
        self.enc2 = ConvBlock(base_channels*2, base_channels*4, stride=2)
        self.bottleneck = ConvBlock(base_channels*4, base_channels*8, stride=1)

        # Decoder with ConvTranspose
        self.up1 = nn.ConvTranspose1d(base_channels*8, base_channels*4, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_channels*8, base_channels*4)

        self.up2 = nn.ConvTranspose1d(base_channels*4, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_channels*4, base_channels*2)

        self.up3 = nn.ConvTranspose1d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_channels*2, base_channels)

        self.final = nn.Sequential(
            nn.Conv1d(base_channels, 1, kernel_size=3, padding=1),
            nn.Tanh()
        )
        self.shortcut = nn.Linear(input_len, output_len)

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        base = self.shortcut(x_flat).view(b, 1, self.output_len)

        feat = self.expander(x_flat).view(b, -1, 16)
        e1 = self.enc1(feat)
        e2 = self.enc2(e1)
        bn = self.bottleneck(e2)

        d1 = self.up1(bn)
        e2_up = F.interpolate(e2, size=d1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e2_up], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        e1_up = F.interpolate(e1, size=d2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e1_up], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        d3 = F.interpolate(d3, size=self.output_len, mode='linear', align_corners=False)
        out = self.final(d3)
        return out + base
