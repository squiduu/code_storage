import torch.nn as nn
from layer_norm import BertLayerNorm
import torch


class Embedding(nn.Module):
    """
    Desc:
        convert the followings to embedded vectors
            1) word ID column of sentence
            2) information of the first or second sentence
    Args:
        config: BERT training configuration file
    """

    def __init__(self, config: dict):
        super().__init__()

        # set token embedding
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        # set positional embedding as learnable
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        # set sentence embedding for NSP i.e., [0, 0, ..., 0, 1, 1, ..., 1]
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        # set layer normalization
        self.layer_norm = BertLayerNorm(config.hidden_size, 1e-12)
        # set dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, token_type_ids=None):
        """
        Args:
            x: token ID vectors, dimension equals [batch_size, seq_len]
            tok_type_ids: sentence ID vectors, dimension equals [batch_size, seq_len]
        """

        # get token embedding for input token ID vectors
        word_embeddings = self.word_embeddings(input_ids)

        # get sentence embedding if it is not None
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # get sequence length
        seq_length = input_ids.size(1)
        # set position ID vectors
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        # set dimension of positional ID vector to input token ID vectors
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # get position embedding
        position_embedddings = self.position_embeddings(position_ids)

        # make total embedding with summation
        embeddings = word_embeddings + position_embedddings + token_type_embeddings
        # get layer normalized embedding
        embeddings = self.layer_norm(embeddings)
        # apply dropout to total embedding
        embeddings = self.dropout(embeddings)

        # dimension equals [batch_size, seq_len, hidden_size]
        return embeddings
