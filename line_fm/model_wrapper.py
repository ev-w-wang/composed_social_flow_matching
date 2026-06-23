import torch
from torch import nn
from flow_matching.utils import ModelWrapper


class FMModelWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, start: torch.Tensor, end: torch.Tensor, x0: torch.Tensor):
        super(FMModelWrapper, self).__init__(model)
        self.start = start
        self.end = end
        self.x0 = x0

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        start = self.start.expand(x.shape[0], -1)
        end = self.end.expand(x.shape[0], -1)
        x0 = self.x0.expand(x.shape[0], -1, -1)
        input = torch.cat([start, end], dim=1)
        return self.model(x0, t, input)
