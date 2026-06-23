import torch
from torch import nn
# from classic.unet import TempUNet


# class FM(TempUNet):
#     def __init__(self, input_dim=4, num_points=10, state_dim=2, hidden_dim=32, dim_mults=(1,2)):
#         super(FM, self).__init__(horizon=num_points, out_dim=num_points*state_dim, cond_dim=input_dim, dim=hidden_dim, dim_mults=dim_mults)
        


class FM(nn.Module):
    def __init__(self, input_dim=4, num_points=10, state_dim=2, hidden_dim=64):
        super(FM, self).__init__()
        self.num_points = num_points
        self.state_dim = state_dim
        noise_dim = num_points * state_dim
        time_dim = 1
        output_dim = num_points * state_dim
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
        )
        self.x0_embed = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x0, t, input):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x_in = torch.cat(
            [self.time_embed(t), self.x0_embed(x0.reshape(x0.shape[0], -1)), self.input_embed(input)],
            dim=1,
        )
        return self.layers(x_in).reshape(x0.shape[0], self.num_points, self.state_dim)
