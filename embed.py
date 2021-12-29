from torch._C import device
import torch.nn as nn
import torch
import math


class WordEmbedder(nn.Module):
    """
    Description:
        give token vectors according to the token ID

    Arguments:
        embedding_vectors: token vectors from token IDs

    Returns:
        an embedding layer with token vectors from input sequence
    """

    def __init__(self, embedding_vectors: torch.Tensor):
        super().__init__()
        # using pre-trained w2v built in torch.nn.Embedding()
        # do not change or update embedding vectors with args freeze equals True
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=embedding_vectors, freeze=True
        )

    def forward(self, x):
        x_embed = self.embeddings(x)

        return x_embed


class PositionalEncoder(nn.Module):
    """
    Description:
        add information that indicates position of input sequences
    """

    def __init__(self, d_model: int = 300, max_seq_len: int = 256):
        super().__init__()

        # dimension of word vectors
        self.d_model = d_model

        # set a empty tensor for positional encoder values
        pe = torch.zeros(size=(max_seq_len, d_model))
        # set a GPU environment
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set to GPU tensor
        pe = pe.to(device)
        # fill positional encoder values
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / d_model)))

        # add a batch dimension as the first dimension
        self.pe = pe.unsqueeze(0)
        # set to stop gradient
        self.pe.requires_grad = False

    def forward(self, x):
        # x equals word embedding vectors
        # multiply sqrt(300) to match since the word embedding is smaller than the positional encoding
        ret = math.sqrt(self.d_model) * x + self.pe

        return ret
