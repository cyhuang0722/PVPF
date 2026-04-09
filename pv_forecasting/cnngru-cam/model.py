"""
双分支 PV 预测模型：Sky CNN+GRU + PV Encoder+GRU → concat → MLP → 4 个 horizon。
输入: sky (B,60,3,H,W), pv_past (B,4,1); 输出: (B,4)，对应 [t+15,t+30,t+45,t+60]。
"""
import torch
import torch.nn as nn


class SkyCNN(nn.Module):
    """单帧天空图 CNN，输出 128 维。"""

    def __init__(self, in_ch=3, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(128, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x):
        # x: (B, C, H, W)
        return self.net(x)


class SkyBranch(nn.Module):
    """60 帧天空图 → CNN 每帧 → GRU → (B, 128)。"""

    def __init__(self, in_ch=3, embed_dim=128, hidden=128):
        super().__init__()
        self.cnn = SkyCNN(in_ch, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)  # (B*T, 128)
        x = x.view(B, T, -1)
        _, h = self.gru(x)
        return h.squeeze(0)


class PVBranch(nn.Module):
    """过去 4 个 PV → Linear(1,32) → GRU(32,64) → (B, 64)。"""

    def __init__(self, hidden=64, pv_proj=32):
        super().__init__()
        self.proj = nn.Linear(1, pv_proj)
        self.gru = nn.GRU(pv_proj, hidden, batch_first=True)

    def forward(self, x):
        # x: (B, 4, 1)
        x = self.proj(x)
        _, h = self.gru(x)
        return h.squeeze(0)


class PVForecastModel(nn.Module):
    """Sky 分支 128 + PV 分支 64 → concat 192 → MLP → 4 (horizon)。"""

    def __init__(self, sky_embed=128, pv_hidden=64, fusion_hidden=128, out_dim=4, dropout=0.2):
        super().__init__()
        self.sky_branch = SkyBranch(in_ch=3, embed_dim=sky_embed, hidden=sky_embed)
        self.pv_branch = PVBranch(hidden=pv_hidden, pv_proj=32)
        fusion_dim = sky_embed + pv_hidden
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, sky, pv_past):
        sky_ctx = self.sky_branch(sky)
        pv_ctx = self.pv_branch(pv_past)
        fusion = torch.cat([sky_ctx, pv_ctx], dim=1)
        return self.head(fusion)
