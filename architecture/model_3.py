import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SpectralTransformer(nn.Module):
    def __init__(self, input_len=9, output_len=1904, d_model=64, num_heads=4, num_layers=2,
                 ff_dim=128, dropout=0.1, latent_len=128):
        super().__init__()
        self.output_len = output_len
        self.latent_len = latent_len
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(input_len, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_len * d_model),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model, max_len=latent_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=output_len / latent_len, mode='linear', align_corners=False),
            nn.Conv1d(d_model, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )

        # Shortcut
        self.shortcut = nn.Linear(input_len, output_len)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)

        # Shortcut
        base = self.shortcut(x_flat).view(b, 1, self.output_len)

        seq = self.input_proj(x_flat).view(b, self.latent_len, self.d_model)
        seq = self.pos_encoder(seq)
        seq = self.transformer(seq)

        seq = seq.transpose(1, 2)
        out = self.upsample(seq)

        return out + base