import torch
import torch.nn as nn
import math
from layer_norm import BertLayerNorm
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    """multi-head self-attention in transformers"""

    def __init__(self, config: dict):
        super().__init__()

        # set the number of attention heads
        self.num_attention_heads = config.num_attention_heads
        # set the dimension of attention heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # set linear layer for query, key, and value
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # set dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor):
        """change shape of tensors to apply attention mechanism"""

        # set temporary tensor size to change dimension for multi-head self-attention
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        # [batch_size, seq_len, hidden_size] -> [batch_size, seq_len, n_heads, head_size]
        x = x.view(*new_x_shape)

        # [batch_size, seq_len, n_heads, head_size] -> [batch_size, n_heads, seq_len, head_size]
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask, attention_show_flg=False
    ):
        """
        Args:
            hidden_states: outuput of previous layer
            attn_mask: byte tensor of padding mask to attention mechanism
            attn_value_flg: flag as to whether to return attention value
        """

        # get query, key, and value tensors with linear layer
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # change shape of query, key, and value tensors
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # apply padding mask
        attention_scores = attention_scores + attention_mask
        # apply softmax
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # apply dropout
        attention_probs = self.dropout(attention_probs)
        # get attention values
        context_layer = torch.matmul(attention_probs, value_layer)

        # restore the dimension of attention values
        # [batch_size, n_heads, seq_len, head_size] -> [bath_size, seq_len, n_heads, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # set temporary tensor size to restore to input dimension
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # [bath_size, seq_len, n_heads, head_size] -> [batch_size, seq_len, hidden_size]
        context_layer = context_layer.view(new_context_layer_shape)

        # determine whether to return the attention probability
        if attention_show_flg is True:
            return context_layer, attention_probs
        else:
            return context_layer


class BertSelfOutput(nn.Module):
    """linear layers and residual connections for output of multi-head self-attention mechanism"""

    def __init__(self, config: dict):
        super().__init__()

        # set linear layer for attention value
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor):
        """
        Args:
            context: output of multi-head self-attention
            residual_input: output of previous layer for residual connection
        """

        # apply linear layer
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # apply layer normalization and residual connection
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BertIntermediate(nn.Module):
    """feed-forward layers in transformer"""

    def __init__(self, config: dict):
        super().__init__()

        # set linear layer for 768 to 3,072 dimension
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: output of attention mechanism
        """

        # apply linear layer
        hidden_states = self.dense(hidden_states)
        # apply gelu function
        hidden_states = F.gelu(hidden_states)

        return hidden_states


class BertAttention(nn.Module):
    """multi-head self-attention layers"""

    def __init__(self, config: dict):
        super().__init__()

        # set multi-head self-attention and attention output
        self.selfattn = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self, input_tensor: torch.Tensor, attention_mask, attention_show_flg=False
    ):
        """
        Args:
            input_tensor: outuput of previous layer
        """

        if attention_show_flg is True:
            self_output, attention_probs = self.selfattn(
                input_tensor, attention_mask, attention_show_flg
            )
            attention_output = self.output(self_output, input_tensor)

            return attention_output, attention_probs

        else:
            self_output = self.selfattn(
                input_tensor, attention_mask, attention_show_flg
            )
            attention_output = self.output(self_output, input_tensor)

            return attention_output


class BertOutput(nn.Module):
    """feed-forward layers in transformer"""

    def __init__(self, config: dict):
        super().__init__()

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        """
        Args:
            hidden_states: output of intermediate feed-forward layer
            attn_outp: output of attention layer
        """

        # apply linear layer
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # apply layer normalization and residual connection
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class BertLayer(nn.Module):
    """transformer block"""

    def __init__(self, config: dict):
        super().__init__()

        # set multi-head self-attention
        self.attention = BertAttention(config)
        # set intermediate layer to handle output of multi-head self-attention
        self.intermediate = BertIntermediate(config)
        # set output layer for linear, layer normalization, and residual connection
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, attention_show_flg=False):
        """
        Args:
            hidden_states: output of embedding layer, [batch_size, seq_len, hidden_size]
        """

        if attention_show_flg is True:
            attention_output, attention_probs = self.attention(
                hidden_states, attention_mask, attention_show_flg
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output, attention_probs

        else:
            attention_output = self.attention(
                hidden_states, attention_mask, attention_show_flg
            )
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)

            return layer_output
