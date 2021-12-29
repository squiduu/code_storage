import torch
import torch.nn as nn
from .attention import MaskedMultiHeadAttention
from .sub_layer import FeedForwardLayer, NormalizedResidualConnection


class Encoder(nn.Module):
    """
    Description:
        encoder = multi-head attention + feed-forward layer + residual connection + layer normalization

    Arguments:
        d_model: dimension of input and output layer
        n_head: number of attention heads
        d_ff_hidden: dimension of feed-forward layer
    """

    def __init__(self, d_model, n_head, d_ff_hidden, dropout: float):
        super().__init__()
        self.attn = MaskedMultiHeadAttention(n_head=n_head, d_model=d_model)
        self.add_norm_layer = NormalizedResidualConnection(
            d_model=d_model, dropout=dropout
        )
        self.ff = FeedForwardLayer(d_model=d_model, d_ff=d_ff_hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask: torch.Tensor):
        y = self.add_norm_layer.forward(
            x, sub_layer=self.attn.forward(query=x, key=x, value=x, mask=mask)
        )
        z = self.add_norm_layer.forward(y, sub_layer=self.ff.forward(y))

        return self.dropout(z)
