import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from config import Config
except ModuleNotFoundError:
    from obstacle_conditioning.config import Config

grid_size = Config["grid_size"]
repulsion_eps = Config.get("repulsion_eps", 1.0)
repulsion_radius = Config.get("repulsion_radius", 20.0)
repulsion_max_magnitude = Config.get("repulsion_max_magnitude", 1.0)


@torch.no_grad()
def plot_vector_field(ax, occupancy, field, step=4, active_only=True, title=""):
    """
    occupancy: (H, W) binary grid
    field: (2, H, W) with field[0]=vx (columns), field[1]=vy (rows)
    """
    occ = occupancy.cpu().numpy()
    vx = field[0].cpu().numpy()
    vy = field[1].cpu().numpy()
    h, w = occ.shape

    ax.imshow(occ, cmap="gray", origin="upper", vmin=0, vmax=1)
    ax.set_xlabel("white = obstacle, black = free")

    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    X, Y = np.meshgrid(xs, ys)
    U = vx[Y, X]
    V = vy[Y, X]

    if active_only:
        magnitude = np.sqrt(U ** 2 + V ** 2)
        mask = magnitude > 1e-3
        U = np.where(mask, U, np.nan)
        V = np.where(mask, V, np.nan)

    ax.quiver(
        X,
        Y,
        U,
        V,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1.0 / step,
        width=0.003,
    )
    ax.set_title(title)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)

@torch.no_grad()
def _sample_obstacle_map(h, w, device):
    occ = torch.zeros(h, w, device=device)
    num_obstacles = random.randint(10, 20)
    for _ in range(num_obstacles):
        if random.random() < 0.5:
            rw = random.randint(max(4, w // 20), max(5, w // 5))
            rh = random.randint(max(4, h // 20), max(5, h // 5))
            x0 = random.randint(0, w - rw)
            y0 = random.randint(0, h - rh)
            occ[y0 : y0 + rh, x0 : x0 + rw] = 1.0
        else:
            cx = random.randint(0, w - 1)
            cy = random.randint(0, h - 1)
            radius = random.randint(max(3, min(h, w) // 30), max(4, min(h, w) // 10))
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
            occ[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = 1.0

    if occ.mean() > 0.4:
        return _sample_obstacle_map(h, w, device)
    return occ


@torch.no_grad()
def generate_occupancy(device, batch_size=1):
    maps = [_sample_obstacle_map(grid_size, grid_size, device) for _ in range(batch_size)]
    return torch.stack(maps).unsqueeze(1)


@torch.no_grad()
def ground_truth_field(occupancy, device):
    """
    Traditional repulsive field: push away from nearest obstacle.
    Free space (within repulsion_radius): magnitude ~ 1 / distance to wall.
    Inside obstacles: magnitude grows with depth (dist_in) for strong escape from interior.
    """
    occ = (occupancy > 0.5).cpu().numpy()
    fields = np.zeros((occ.shape[0], 2, grid_size, grid_size), dtype=np.float32)

    for b in range(occ.shape[0]):
        occ_bin = occ[b, 0].astype(bool)
        dist = distance_transform_edt(~occ_bin).astype(np.float32)
        gy, gx = np.gradient(dist)

        denom = dist + repulsion_eps
        fx = gx / denom
        fy = gy / denom

        if occ_bin.any():
            dist_in = distance_transform_edt(occ_bin).astype(np.float32)
            giy, gix = np.gradient(dist_in)
            inside = occ_bin
            # Inside obstacles: scale by dist_in (not 1/dist_in) so push is stronger
            # at the center than near the boundary — helps escape deep samples.
            fx[inside] = -gix[inside] * (dist_in[inside] + repulsion_eps)
            fy[inside] = -giy[inside] * (dist_in[inside] + repulsion_eps)

        far = (dist > repulsion_radius) & ~occ_bin
        fx[far] = 0.0
        fy[far] = 0.0

        magnitude = np.sqrt(fx ** 2 + fy ** 2)
        clip = magnitude > repulsion_max_magnitude
        fx[clip] *= repulsion_max_magnitude / magnitude[clip]
        fy[clip] *= repulsion_max_magnitude / magnitude[clip]

        fields[b, 0] = fx
        fields[b, 1] = fy

    return torch.from_numpy(fields).to(device=device, dtype=occupancy.dtype)


def repulsion_loss_mask(ground_truth):
    return (ground_truth.pow(2).sum(dim=1, keepdim=True) > 1e-8).float()


def sample_field_at_points(field, points, grid_size=None):
    """
    field: (B, 2, H, W)
    points: (B, num_points, 2) in grid coordinates [0, grid_size)
    returns: (B, num_points, 2) sampled (vx, vy) at each waypoint
    """
    if grid_size is None:
        grid_size = field.shape[-1]
    grid = points / (grid_size - 1) * 2 - 1
    grid = grid.flip(-1).unsqueeze(2)
    sampled = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(-1).permute(0, 2, 1)


@torch.no_grad()
def ground_truth_repulsion_at_points(occupancy, points, device):
    """
    Analytic obstacle composer: compute repulsion GT field and sample at waypoint locations.
    occupancy: (B, 1, H, W) or (B, H, W)
    points: (B, num_points, 2)
    returns: (B, num_points, 2)
    """
    field = ground_truth_field(occupancy, device)
    if field.shape[0] == 1 and points.shape[0] > 1:
        field = field.expand(points.shape[0], -1, -1, -1)
    return sample_field_at_points(field, points, grid_size)


class AnalyticRepulsionField:
    """Precomputed GT repulsion field for use at inference / composition (no learned network)."""

    def __init__(self, occupancy, device):
        self.grid_size = grid_size
        self.device = device
        if occupancy.dim() == 3:
            occupancy = occupancy.unsqueeze(1)
        self.occupancy = occupancy
        self.field = ground_truth_field(occupancy, device)

    @torch.no_grad()
    def at_points(self, points):
        """points: (B, num_points, 2) -> (B, num_points, 2)"""
        field = self.field
        if field.shape[0] == 1 and points.shape[0] > 1:
            field = field.expand(points.shape[0], -1, -1, -1)
        return sample_field_at_points(field, points, self.grid_size)

    @torch.no_grad()
    def __call__(self, points):
        return self.at_points(points)


if __name__ == "__main__":
    occupancy = generate_occupancy(device="cpu", batch_size=1)
    ground_truth = ground_truth_field(occupancy, device="cpu")
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    plot_vector_field(axes, occupancy[0, 0], ground_truth[0], title="Ground truth")
    plt.tight_layout()
    plt.savefig("ground_truth.png")
    plt.show()