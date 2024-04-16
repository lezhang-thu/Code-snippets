# Sample code for prompt-tuning
# transfer HuggingFace BertModel to a `prompted` BertMode

from typing import Optional
import torch
import torch.nn as nn
import transformers
from transformers import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class PromptedBert(BertModel):
    # call it via `x = PromptedBert(bert.config, bert.pooler is not None)`
    # print(x.load_state_dict(bert.state_dict(), strict=False))
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        num_tokens = 10
        self.num_tokens = num_tokens  # number of prompted tokens
        self.prompt_dropout = nn.Dropout(0.1)
        prompt_dim = 768

        self.deep_prompt_embeddings = nn.Parameter(
            torch.randn(config.num_hidden_layers, num_tokens, prompt_dim))

        ## xavier_uniform initialization
        ## 170? - reduce(mul, patch_size, 1) where 1 initializer
        #val = math.sqrt(6. / float(prompt_dim))  # noqa
        #nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, x):
        return torch.cat(
            (x[:, :1, :],
             self.prompt_dropout(self.deep_prompt_embeddings[0].expand(
                 x.shape[0], -1, -1)), x[:, 1:, :]),
            dim=1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        x = torch.ones(((batch_size, self.num_tokens)), device=device)
        attention_mask = torch.cat([x, attention_mask], dim=1)

        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :
                                                                     seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape,
                                         dtype=torch.long,
                                         device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # debug
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length + self.num_tokens))

        x = self.embeddings(
            input_ids=input_ids,
            position_ids=None,
            token_type_ids=token_type_ids,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)
        hidden_states = None

        # debug - start
        for i, layer_module in enumerate(self.encoder.layer):
            if i == 0:
                layer_outputs = layer_module(
                    embedding_output,
                    extended_attention_mask,
                )
            else:
                deep_prompt_emb = self.prompt_dropout(
                    self.deep_prompt_embeddings[i].expand(batch_size, -1, -1))

                hidden_states = torch.cat(
                    (hidden_states[:, :1, :], deep_prompt_emb,
                     hidden_states[:, 1 + self.num_tokens:, :]),
                    dim=1)
                layer_outputs = layer_module(
                    hidden_states,
                    extended_attention_mask,
                )
            hidden_states = layer_outputs[0]

        sequence_output = hidden_states
        pooled_output = self.pooler(
            sequence_output) if self.pooler is not None else None
        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class XModel(torch.nn.Module):

    def __init__(self,
                 emb_dim,
                 domain_num):
        super(XDModel, self).__init__()
        self.domain_num = domain_num
        self.emb_dim = emb_dim
        self.bert = BertModel.from_pretrained(
            './bert-chinese-bert-wwm-ext/').requires_grad_(False)

        # debug
        x = PromptedBert(self.bert.config, self.bert.pooler is not None)
        print(x.load_state_dict(self.bert.state_dict(), strict=False))
        for _ in [x.embeddings, x.encoder, x.pooler]:
            _.requires_grad_(False)
        del self.bert
        self.bert = x

        self.classifier = nn.Sequential(nn.Linear(768, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(0.1),
                                        nn.Linear(128, 2))
        self.classifier.apply(init_weights)
