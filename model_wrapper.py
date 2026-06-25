import torch
from torch import nn
from flow_matching.utils import ModelWrapper
from obstacle_conditioning.network import sample_field_at_points
from obstacle_conditioning.occupancy_utils import AnalyticRepulsionField


class ComposedModelWrapper(ModelWrapper):
    def __init__(
        self,
        line_fm_model: nn.Module,
        obstacle_model: nn.Module,
        input: torch.Tensor,
        x0: torch.Tensor,
        occupancy: torch.Tensor,
        alpha: float,
    ):
        super().__init__(line_fm_model)
        self.line_fm_model = line_fm_model
        self.grid_size = obstacle_model.grid_size
        self.input = input
        self.x0 = x0
        self.alpha = alpha
        with torch.no_grad():
            self.obstacle_field = obstacle_model.forward_field(occupancy)

    def _format_time(self, t: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return t

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        batch_size = x.shape[0]
        t = self._format_time(t, batch_size, x.device, x.dtype)
        x0 = self.x0.expand(batch_size, -1, -1)
        input = self.input.expand(batch_size, -1)

        line_output = self.line_fm_model(x0, t, input)

        field = self.obstacle_field
        if field.shape[0] == 1:
            field = field.expand(batch_size, -1, -1, -1)
        obstacle_output = sample_field_at_points(field, x, self.grid_size)

        return line_output + self.alpha * obstacle_output

class ComposedModelWrapperWithRepulsionField(ModelWrapper):
    def __init__(
        self,
        line_fm_model: nn.Module,
        repulsion_field: AnalyticRepulsionField ,
        input: torch.Tensor,
        x0: torch.Tensor,
        alpha: float,
    ):
        super().__init__(line_fm_model)
        self.line_fm_model = line_fm_model
        self.repulsion_field = repulsion_field
        self.input = input
        self.x0 = x0
        self.alpha = alpha

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        batch_size = x.shape[0]
        t = self._format_time(t, batch_size, x.device, x.dtype)
        x0 = self.x0.expand(batch_size, -1, -1)
        input = self.input.expand(batch_size, -1)

        line_output = self.line_fm_model(x0, t, input)
        obstacle_output = self.repulsion_field.at_points(x)
        # line_out_mag = torch.norm(line_output, dim=1)
        # obstacle_out_mag = torch.norm(obstacle_output, dim=1)
        # print(f"Line output magnitude: {line_out_mag.mean().item():.4f}, Obstacle output magnitude: {obstacle_out_mag.mean().item():.4f}")
        return line_output + self.alpha * obstacle_output

    def _format_time(self, t: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device, dtype=dtype)
        if t.dim() == 0:
            t = t.expand(batch_size)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return t