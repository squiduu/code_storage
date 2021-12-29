import torch
from embedding import PositionalEncoder, WordEmbedder
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    """
    Description:
        implement only a single-head attention for easy practice
    """

    def __init__(self, d_model: int = 300, n_heads: int = 1):
        super().__init__()

        # set linear layers for query, key, and value
        self.q_lin = nn.Linear(in_features=d_model, out_features=d_model)
        self.k_lin = nn.Linear(in_features=d_model, out_features=d_model)
        self.v_lin = nn.Linear(in_features=d_model, out_features=d_model)
        # set a linear layer for concatenated output
        self.out_lin = nn.Linear(in_features=d_model, out_features=d_model)

        # set a scaling factor for scaled-dot attention
        self.d_k = d_model / n_heads

    def forward(self, q, k: torch.Tensor, v, mask: torch.Tensor):
        # set query, key, and value respectively
        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)

        # get attention score
        attn = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        # set mask for padding mask of attention mechanism
        mask = mask.unsqueeze(1)
        # apply padding mask to attention score
        attn = attn.masked_fill(mask=(mask == 0), value=-1e9)
        # apply softmax to the last dimension of attention score
        attn = F.softmax(input=attn, dim=-1)
        # get attention value matrices
        attn_value = torch.matmul(attn, v)

        # get concatenated output
        output = self.out_lin(attn_value)

        return output, attn_value


class FeedForward(nn.Module):
    """
    Description:
        a fully connected layer after the attention layer
    """

    def __init__(self, d_model, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()

        # set a fully connected layer structure
        self.lin_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.dropout = nn.Dropout(p=dropout)
        self.lin_2 = nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x):
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        output = self.lin_2(x)

        return output


class TransformerBlock(nn.Module):
    """
    Description:
        layer norm -> attention -> dropout -> residual conn
        -> layer norm -> feed forward -> dropout -> residual conn
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # set layer normalizations
        self.layernorm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.layernorm_2 = nn.LayerNorm(normalized_shape=d_model)
        # set attention layer
        self.attn = Attention(d_model=d_model, n_heads=1)
        # set feed forward layer
        self.ff = FeedForward(d_model=d_model, d_ff=1024, dropout=dropout)
        # set dropout layers
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        """layer norm -> attention -> dropout -> residual conn"""
        x = self.layernorm_1(x)
        x, attn_value = self.attn(x, x, x, mask)
        output = self.dropout_1(x)
        y = x + output

        """layer norm -> feed forward -> dropout -> residual conn"""
        y = self.layernorm_2(y)
        y = self.ff(y)
        output = self.dropout_2(y)
        z = y + output

        return z, attn_value


class ClassificationLayer(nn.Module):
    """
    Description:
        set classification layer to positive and negative after transformer block
    """

    def __init__(self, d_model=300, d_output=2):
        super().__init__()

        # set a fully connected layer
        self.lin = nn.Linear(in_features=d_model, out_features=d_output)

        # initialize mean and standard deviation of the fully connected layer
        nn.init.normal_(tensor=self.lin.weight, mean=0, std=0.02)
        nn.init.normal_(tensor=self.lin.bias, mean=0)

    def forward(self, x):
        # get the word representation
        x = x[:, 0, :]
        output = self.lin(x)

        return output


class TransformerClassification(nn.Module):
    """
    Description:
        word embedding -> positional embedding -> transformer block 1
        -> transformer block 2 -> classification layer
    """

    def __init__(self, embed_vec, d_model=300, max_seq_len=256, d_output=2):
        super().__init__()

        # make model architecture
        self.net_1 = WordEmbedder(embedding_vectors=embed_vec)
        self.net_2 = PositionalEncoder(d_model=d_model, max_seq_len=max_seq_len)
        self.net_3_1 = TransformerBlock(d_model=d_model, dropout=0.1)
        self.net_3_2 = TransformerBlock(d_model=d_model, dropout=0.1)
        self.net_4 = ClassificationLayer(d_model=d_model, d_output=d_output)

    def forward(self, x, mask):
        # get word embedding vectors from input sequence
        x = self.net_1(x)
        # add positional information
        x = self.net_2(x)
        # go through transformer block 1 & 2
        x, attn_value_1 = self.net_3_1(x, mask)
        x, attn_value_2 = self.net_3_2(x, mask)
        y = self.net_4(x)

        return y, attn_value_1, attn_value_2
