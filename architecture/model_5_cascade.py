import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class SpectralResNet(nn.Module):
    def __init__(
        self,
        input_len=9,
        output_len=1904,
        hidden_dim=512,
        latent_len=32,
        dropout=0.1
    ):
        super().__init__()

        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.latent_len = latent_len

        self.encoder = nn.Sequential(
            nn.Linear(input_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.seed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * latent_len),
            nn.GELU()
        )

        # Latent conv blocks
        self.latent_blocks = nn.Sequential(
            ResidualBlock1D(hidden_dim, dropout),
            ResidualBlock1D(hidden_dim, dropout),
            ResidualBlock1D(hidden_dim, dropout),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(size=128, mode='linear', align_corners=False),
            nn.Conv1d(hidden_dim, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Upsample(size=512, mode='linear', align_corners=False),
            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Upsample(size=1024, mode='linear', align_corners=False),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Upsample(size=output_len, mode='linear', align_corners=False),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.GELU(),

            nn.Conv1d(32, 1, kernel_size=1)
        )

        self.refine = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=1)
        )

        # shortcut
        self.shortcut = nn.Linear(input_len, output_len)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)

        # baseline
        base = self.shortcut(x_flat).unsqueeze(1)

        # encode
        feat = self.encoder(x_flat)

        # seed sequence
        feat = self.seed(feat)
        feat = feat.view(x.size(0), self.hidden_dim, self.latent_len)

        # latent processing
        feat = self.latent_blocks(feat)

        # decode
        out = self.decoder(feat)

        # refine residual
        details = self.refine(out)

        return base + out + details