from network import FM
import torch
import matplotlib.pyplot as plt
from config import Config
from flow_matching.solver import ODESolver
from model_wrapper import FMModelWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = Config["input_dim"]
state_dim = Config["state_dim"]
num_points = Config["num_points"]
hidden_dim = Config["hidden_dim"]
n_samples = Config["n_samples"]

model = FM(input_dim=input_dim, num_points=num_points, state_dim=state_dim, hidden_dim=hidden_dim)
model.load_state_dict(torch.load("model_final.pth", map_location=device))
model.to(device)
model.eval()

# start = torch.randn(1, state_dim, device=device)
# end = torch.randn(1, state_dim, device=device)
start = torch.rand(1, input_dim//2, device=device) * 128
end = torch.rand(1, input_dim//2, device=device) * 128
noise = torch.randn(n_samples, num_points, state_dim, device=device)

time_grid = torch.linspace(0, 1, 1000, device=device)
solver = ODESolver(FMModelWrapper(model, start, end, noise))
solution = solver.sample(
    x_init=noise,
    time_grid=time_grid,
    step_size=0.001,
    method="midpoint",
    return_intermediates=True,
)

trajectories = solution[-1].detach().cpu()
for i in range(n_samples):
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], "o-", alpha=0.6)
plt.plot(
    [start[0, 0].cpu(), end[0, 0].cpu()],
    [start[0, 1].cpu(), end[0, 1].cpu()],
    "k--",
    linewidth=2,
    label="start/end",
)
plt.legend()
plt.show()
