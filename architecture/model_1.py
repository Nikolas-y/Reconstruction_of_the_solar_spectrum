import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(self.linear1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return residual + self.dropout(x)

class SpectralGenerator(nn.Module):
    def __init__(self, input_len=9, output_len=1904, hidden_dim=512, n_blocks=4, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_len, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )
        self.shortcut = nn.Linear(input_len, output_len)

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        base = self.shortcut(x_flat)
        feat = self.input_layer(x_flat)
        for blk in self.res_blocks:
            feat = blk(feat)
        detail = self.output_layer(feat)
        return (base + detail).unsqueeze(1)

# govnina peredelat nado
# ahaha norm prosto ne dlya etoi zadachi
class SpectralDiscriminator(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=64, n_layers=3):
        super().__init__()
        layers = []
        in_ch = input_channels
        out_ch = hidden_channels
        for i in range(n_layers):
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        layers.append(
            nn.Conv1d(in_ch, 1, kernel_size=4, stride=1, padding=1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class SpectralGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.gen = generator
        self.disc = discriminator

    def forward(self, x):
        return self.gen(x)
