import torch.nn as nn
from transformer import BertLayer


class BertEncoder(nn.Module):
    """make encoder with repeated transformer blocks"""

    def __init__(self, config: dict):
        super().__init__()

        # stack transformer blocks repeatedly
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        attention_show_flg=False,
    ):
        """
        Args:
            hidden_states: output of embedding layer
            all_layers_flg: whether to get output of all the layers or not,
                it provides that you can see how the attention is applied to each layer
        """

        # set output list
        all_encoder_layers = []

        # for each layer
        for layer_module in self.layer:
            # in case of visualizing attention map
            if attention_show_flg is True:
                # get output of transformer block
                hidden_states, attention_probs = layer_module(
                    hidden_states, attention_mask, attention_show_flg
                )
            else:
                hidden_states = layer_module(
                    hidden_states, attention_mask, attention_show_flg
                )

            # in case of using outputs of all layers
            if output_all_encoded_layers is True:
                all_encoder_layers.append(hidden_states)

        # in case of using output of the last layer
        if output_all_encoded_layers is False:
            all_encoder_layers.append(hidden_states)

        # in case of visualizing attention map
        if attention_show_flg is True:
            return all_encoder_layers, attention_probs
        else:
            return all_encoder_layers
