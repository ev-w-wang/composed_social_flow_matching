from network import ObstacleConditioning
import torch
from config import Config
import time
from occupancy_utils import generate_occupancy, ground_truth_field, repulsion_loss_mask

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


if __name__ == "__main__":
    curr_time = time.time()
    for step in range(num_steps):
        occupancy = generate_occupancy(device, batch_size=batch_size)
        ground_truth = ground_truth_field(occupancy, device)
        output = model.forward_field(occupancy)
        loss_mask = repulsion_loss_mask(ground_truth)
        loss = torch.sum((output - ground_truth) ** 2 * loss_mask) / loss_mask.sum().clamp(min=1)
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
                print(f"Model saved at step 8000: model_8000.pth")
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    torch.save(model.state_dict(), "model_final.pth")
    print(f"Model saved at step {num_steps}: model_final.pth")
