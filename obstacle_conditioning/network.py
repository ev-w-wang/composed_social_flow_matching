import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.block(x)


class OccupancyFieldEncoder(nn.Module):
    """Embed occupancy grid and decode to a 2D velocity field."""

    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_dim, stride=1)
        self.enc2 = ConvBlock(base_dim, base_dim * 2, stride=2)
        self.enc3 = ConvBlock(base_dim * 2, base_dim * 4, stride=2)
        self.enc4 = ConvBlock(base_dim * 4, base_dim * 4, stride=2)

        self.up3 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(base_dim * 6, base_dim * 2)
        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_dim * 3, base_dim)
        self.head = nn.Conv2d(base_dim, 2, 3, padding=1)

    def forward(self, occupancy):
        # occupancy: (B, 1, H, W), values in {0, 1} or [0, 1]
        input_size = occupancy.shape[-2:]
        e1 = self.enc1(occupancy)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.enc4(e3)

        d3 = self.up3(bottleneck)
        d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=True)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=True)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        field = self.head(d2)
        if field.shape[-2:] != input_size:
            field = F.interpolate(field, size=input_size, mode="bilinear", align_corners=True)
        return field


def sample_field_at_points(field, points, grid_size):
    """
    field: (B, 2, H, W)
    points: (B, num_points, 2) in grid coordinates [0, grid_size)
    returns: (B, num_points, 2)
    """
    grid = points / (grid_size - 1) * 2 - 1
    grid = grid.flip(-1).unsqueeze(2)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


class ObstacleConditioning(nn.Module):
    def __init__(self, grid_size=500, num_points=10, state_dim=2, hidden_dim=128, base_dim=32):
        super().__init__()
        self.grid_size = grid_size
        self.num_points = num_points
        self.state_dim = state_dim

        self.grid_encoder = OccupancyFieldEncoder(in_channels=1, base_dim=base_dim)

    def forward_field(self, occupancy):
        """occupancy (B, H, W) or (B, 1, H, W) -> full field (B, 2, H, W)"""
        if occupancy.dim() == 3:
            occupancy = occupancy.unsqueeze(1)
        return self.grid_encoder(occupancy)

    def forward(self, occupancy, points):
        """
        occupancy: (B, H, W) binary grid
        points: (B, num_points, 2) query locations for composition with line_fm
        returns: (B, num_points, 2) obstacle repulsion velocity at each point
        """
        field = self.forward_field(occupancy)
        return sample_field_at_points(field, points, self.grid_size)
