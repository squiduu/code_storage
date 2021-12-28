import torch.nn as nn
import torch


class BertLayerNorm(nn.Module):
    """layer normalization"""

    def __init__(self, hidden_size: int, eps=1e-12):
        super().__init__()

        # set gamma and beta of layer normalization as learnable parameters
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.ones(hidden_size))
        # set epsilon to avoid being divided by zero
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor):
        u = x.mean(dim=-1, keepdim=True)
        s = (x - u).pow(2).mean(dim=-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)

        return self.gamma * x + self.beta
