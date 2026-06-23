import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Config = {
    "num_steps": int(5e4),
    "batch_size": 256,
    "lr": 1e-3,
    "input_dim": 4,
    "state_dim": 2,
    "num_points": 10,
    "hidden_dim": 64,
    "n_samples": 16,
    "step_size": 0.001,
}