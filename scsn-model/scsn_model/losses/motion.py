from __future__ import annotations

import torch
import torch.nn.functional as F


def warp_feature_map(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    bsz, ch, h, w = feature.shape
    device = feature.device
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(bsz, 1, 1, 1)

    dx = flow[:, 0] * (2.0 / max(w - 1, 1))
    dy = flow[:, 1] * (2.0 / max(h - 1, 1))
    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] + dx
    grid[..., 1] = grid[..., 1] + dy
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="border", align_corners=True)


def motion_regularization_loss(
    frame_features: torch.Tensor,
    motion_fields: torch.Tensor,
    warp_weight: float = 1.0,
    smooth_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    warp_loss = torch.tensor(0.0, device=frame_features.device)
    smooth_loss = torch.tensor(0.0, device=frame_features.device)

    n_pairs = motion_fields.shape[1]
    for idx in range(n_pairs):
        prev_feat = frame_features[:, idx]
        curr_feat = frame_features[:, idx + 1]
        flow = motion_fields[:, idx]
        warped = warp_feature_map(prev_feat, flow)
        warp_loss = warp_loss + F.l1_loss(warped, curr_feat)

        dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        smooth_loss = smooth_loss + dx.abs().mean() + dy.abs().mean()

    warp_loss = warp_loss / max(n_pairs, 1)
    smooth_loss = smooth_loss / max(n_pairs, 1)
    total = warp_weight * warp_loss + smooth_weight * smooth_loss
    return total, {
        "warp_loss": float(warp_loss.detach().cpu()),
        "smooth_loss": float(smooth_loss.detach().cpu()),
    }

