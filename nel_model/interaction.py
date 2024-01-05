
from typing import Any, Optional, Tuple
import math

import torch
from torch import nn, Tensor, device

from transformers.activations import ACT2FN
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
)
# from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
# from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn.functional as F


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        qks = (key_states, value_states) if output_qks else None

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, qks


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size    # 768

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        mixed_query_layer = self.query(hidden_states)


        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        qks = (key_layer, value_layer) if output_qks else None

        if past_key_values is not None:
            key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_values[0], value_layer], dim=2)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None and past_key_values is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            bsz, nheads, length, dsize = past_key_values[0].size()
            visual_attention_mask = torch.ones((bsz, 1, 1, length)).to(attention_mask.device)  # bsz, 12, len, 64
            attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)
            attention_scores = attention_scores + attention_mask
        elif attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)    # bsz, 128, 768

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs, qks


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        self_outputs, qks = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            visual_hidden_state,
            output_qks,
            current_layer,
            past_key_values,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs, qks


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.att = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        past_key_values: torch.Tensor = None,
        current_layer: int = None,
        output_qks = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        choice = False
        if choice:
            residual = hidden_states

            hidden_states = self.layer_norm1(hidden_states)

            bsz, tgt_len, embed_dim = hidden_states.size()
            proj_shape = (bsz, -1, self.embed_dim)
            past_key_values[0] = past_key_values[0].view(*proj_shape)
            past_key_values[1] = past_key_values[1].view(*proj_shape)
            hidden_states, _ = self.att(hidden_states, past_key_values[0], past_key_values[1])  # 8, 197, 768

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = hidden_states

            return outputs
        else:
            residual = hidden_states
            print(hidden_states)
            hidden_states = self.layer_norm1(hidden_states)
            
            hidden_states, attn_weights, qks = self.self_attn(
                hidden_states=hidden_states,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                output_qks=output_qks,
                current_layer=current_layer,
            )
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = self.layer_norm2(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (attn_weights,)

            if output_qks:
                outputs += (qks, )

            return outputs

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.att = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        visual_hidden_state=None,
        output_qks=None,
        current_layer=None,
        past_key_values=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None

        # proj_shape = (hidden_states.size(0), -1, self.embed_dim)
        # past_key_values[0] = past_key_values[0].view(*proj_shape)
        # past_key_values[1] = past_key_values[1].view(*proj_shape)

        # hidden_states, _ = self.att(hidden_states, past_key_values[0], past_key_values[1])  # 8, 60, 768

        self_attention_outputs, qks = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            visual_hidden_state=visual_hidden_state,
            output_qks=output_qks,
            current_layer=current_layer,
            past_key_values=past_key_values,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        if output_qks:
            outputs += (qks,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    

    
class Generator(nn.Module):
    def __init__(self, config, query_num):
        super().__init__()
        self.query_num = query_num

        self.modality_query = torch.nn.Parameter(torch.randn(self.query_num, config.hidden_size), requires_grad=True)
        generator_layer = TransformerDecoderLayer(config.hidden_size,config.num_attention_heads)
        self.generator_layer = TransformerDecoder(generator_layer, 1)


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.modality_query.data.uniform_(-initrange, initrange)

    def forward(self,
        hidden_states,
        attention_mask=None,
    ):
        bsz,max_len, _ = hidden_states.size()
        if attention_mask is not None:
            attention_mask = (attention_mask.squeeze(1).squeeze(1)==0).sum(1)
            src_mask = (
            (1.0 - get_mask(attention_mask, max_len))
                .type(torch.bool)
                .to(hidden_states.device)
            )

        else:
            src_mask = None
        modality_query = self.modality_query.unsqueeze(0).repeat(bsz, 1, 1)
        encoder_output = hidden_states.transpose(0,1)
        modality_query = modality_query.transpose(0,1)
        output = self.generator_layer(memory=encoder_output, tgt = modality_query, memory_key_padding_mask = src_mask)

        return output.transpose(0,1)
