from network import ObstacleConditioning
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from occupancy_utils import generate_occupancy, ground_truth_field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

grid_size = Config["grid_size"]
num_points = Config["num_points"]
state_dim = Config["state_dim"]
hidden_dim = Config["hidden_dim"]
base_dim = Config["base_dim"]

model = ObstacleConditioning(grid_size=grid_size, num_points=num_points, state_dim=state_dim, hidden_dim=hidden_dim, base_dim=base_dim)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.to(device)
model.eval()

occupancy = generate_occupancy(device, batch_size=1)
ground_truth = ground_truth_field(occupancy, device)
output = model.forward_field(occupancy)
obstacle_mask = (occupancy > 0.5).float()
loss = torch.sum((output - ground_truth) ** 2 * obstacle_mask) / obstacle_mask.sum().clamp(min=1)
print(f"Loss: {loss.item():.4f}")

@torch.no_grad()
def plot_vector_field(ax, occupancy, field, step=4, obstacle_only=True, title=""):
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

    if obstacle_only:
        mask = occ[Y, X] > 0.5
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


occ = occupancy[0, 0]
pred = output[0]
gt = ground_truth[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_vector_field(axes[0], occ, pred, step=4, title="Predicted")
plot_vector_field(axes[1], occ, gt, step=4, title="Ground truth")
plt.tight_layout()
plt.savefig("test.png")
plt.show()