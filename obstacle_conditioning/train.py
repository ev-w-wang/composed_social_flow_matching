from network import ObstacleConditioning
import torch
from config import Config
import time
import random
import numpy as np
from scipy.ndimage import distance_transform_edt

num_steps = Config["num_steps"]
batch_size = Config["batch_size"]
lr = Config["lr"]
grid_size = Config["grid_size"]
num_points = Config["num_points"]
state_dim = Config["state_dim"]
hidden_dim = Config["hidden_dim"]
base_dim = Config["base_dim"]

model = ObstacleConditioning(grid_size=grid_size, num_points=num_points, state_dim=state_dim, hidden_dim=hidden_dim, base_dim=base_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
start_time = time.time()
total_loss = 0


@torch.no_grad()
def _sample_obstacle_map(h, w, device):
    occ = torch.zeros(h, w, device=device)
    num_obstacles = random.randint(2, 6)
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
def generate_occupancy():
    maps = [_sample_obstacle_map(grid_size, grid_size, device) for _ in range(batch_size)]
    return torch.stack(maps).unsqueeze(1)


@torch.no_grad()
def ground_truth_field(occupancy):
    # Euclidean distance transform is CPU-bound in scipy; keep grid_size modest (e.g. 128).
    occ = (occupancy > 0.5).cpu().numpy()
    fields = np.zeros((occ.shape[0], 2, grid_size, grid_size), dtype=np.float32)
    for b in range(occ.shape[0]):
        dist = distance_transform_edt(occ[b, 0])
        gy, gx = np.gradient(dist)
        obstacle_mask = occ[b, 0] > 0
        # Gradient of distance-to-free increases toward obstacle interior; negate to push out.
        fields[b, 0, obstacle_mask] = -gx[obstacle_mask]
        fields[b, 1, obstacle_mask] = -gy[obstacle_mask]

        magnitude = np.sqrt(fields[b, 0] ** 2 + fields[b, 1] ** 2)
        valid = obstacle_mask & (magnitude > 1e-8)
        fields[b, 0, valid] /= magnitude[valid]
        fields[b, 1, valid] /= magnitude[valid]

    return torch.from_numpy(fields).to(device=device, dtype=occupancy.dtype)


if __name__ == "__main__":
    curr_time = time.time()
    for step in range(num_steps):
        occupancy = generate_occupancy()
        ground_truth = ground_truth_field(occupancy)
        output = model.forward_field(occupancy)
        obstacle_mask = (occupancy > 0.5).float()
        loss = torch.sum((output - ground_truth) ** 2 * obstacle_mask) / obstacle_mask.sum().clamp(min=1)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (step + 1) % 500 == 0:
            avg_loss = total_loss / 500
            print(f"Step {step + 1}, Loss {avg_loss}, Time taken: {time.time() - curr_time} seconds")
            curr_time = time.time()
            total_loss = 0
            if step + 1 == 8000:
                torch.save(model.state_dict(), "model_8000.pth")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    torch.save(model.state_dict(), "model_final.pth")
