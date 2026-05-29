import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, dim, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.inner_dim = int(expand * dim)
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim,
            out_channels=self.inner_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.inner_dim
        )
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.05)

    def forward(self, x):
        res = x
        x = self.norm(x)
        b, l, d = x.shape
        x_and_gate = self.in_proj(x)
        x, gate = x_and_gate.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :l]
        x = x.transpose(1, 2)
        x = F.silu(x)
        x = x * torch.sigmoid(gate)
        return res + self.dropout(self.out_proj(x))

class LightSpectralMamba(nn.Module):
    def __init__(self, input_len=9, output_len=1904, d_model=128, n_layers=6):
        super().__init__()
        self.output_len = output_len
        self.d_model = d_model
        self.latent_len = 64

        self.embedding = nn.Sequential(
            nn.Linear(input_len, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, self.latent_len * d_model),
            nn.GELU()
        )

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model // 2, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.ConvTranspose1d(d_model // 2, d_model // 4, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(d_model // 4),
            nn.GELU(),
            nn.Conv1d(d_model // 4, 1, kernel_size=3, padding=1)
        )
        self.out_adapter = None
        self._create_adapter()
        self.base_regressor = nn.Linear(input_len, output_len)
        self._init_weights()

    def _create_adapter(self):
        with torch.no_grad():
            dummy_x = torch.zeros(1, self.d_model, self.latent_len)
            dummy_out = self.upsample(dummy_x)
            self.out_adapter = nn.Linear(dummy_out.shape[-1], self.output_len)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        b = x.shape[0]
        base = self.base_regressor(x).view(b, 1, self.output_len)
        x_lat = self.embedding(x).view(b, self.latent_len, self.d_model)
        for layer in self.mamba_layers:
            x_lat = layer(x_lat)
        x_up = x_lat.transpose(1, 2)
        x_up = self.upsample(x_up)
        out = self.out_adapter(x_up)
        return out + base