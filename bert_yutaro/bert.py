import torch
import torch.nn as nn
from embedding import Embedding
from encoder import BertEncoder
from pooler import BertPooler


class BertModel(nn.Module):
    """make BERT model with connecting all modules"""

    def __init__(self, config: dict):
        super().__init__()

        # set modules
        self.embeddings = Embedding(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        attention_show_flg=False,
    ):
        """
        Args:
            inp_ids: token ID vectors, [batch_size, seq_len]
            sent_ids: sentence ID vectors, [batch_size, seq_len]
        """

        # make meaningless tensor if sentence ID ventor is not assigned
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        # make meaningless tensor if attention mask is not assigned
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # set extended attention mask for multi-head self-attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # forward propagation
        # get output of embedding layers
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # get output of transformer layers
        if attention_show_flg is True:
            encoded_layers, attention_probs = self.encoder(
                embedding_output,
                extended_attention_mask,
                output_all_encoded_layers,
                attention_show_flg,
            )
        else:
            encoded_layers = self.encoder(
                embedding_output,
                extended_attention_mask,
                output_all_encoded_layers,
                attention_show_flg,
            )

        # get output of pooler with the last layer
        pooler_output = self.pooler(encoded_layers[-1])

        # get encoder output as tensor, not list if flag is False
        if output_all_encoded_layers is False:
            encoded_layers = encoded_layers[-1]

        if attention_show_flg is True:
            return encoded_layers, pooler_output, attention_probs
        else:
            return encoded_layers, pooler_output
