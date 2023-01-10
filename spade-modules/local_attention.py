import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Dict, List, Optional, Tuple

from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise


def _pad_to_multiple(x: torch.Tensor, block_len: int, dim: int, pad_value: int = 0) -> torch.Tensor:
    """Pad a tensor so that a sequence length will be a multiple of `block_len`"""
    pad_len = -x.shape[dim] % block_len
    # Handle cases when an empty input sequence is given
    if not all(x.shape):
        new_shape = list(x.shape)
        new_shape[dim] += pad_len
        return torch.zeros(new_shape, dtype=x.dtype)

    pad = [(0, 0)] * x.ndim
    pad[dim] = (0, pad_len)
    pad = sum(pad[::-1], ())
    x = F.pad(x, pad=pad, mode="constant", value=pad_value)
    return x


def _split_into_blocks(x: torch.Tensor, block_len: int, dim: int) -> torch.Tensor:
    """Split an input tensor into blocks of a given `block_len` along the given `dim`. If the dimension length
    is not a multiple of `block_len`, it will be padded first with selected `pad_value`.
    """
    # pad tensor to multiple of block_len
    if x.shape[dim] % block_len != 0:
        x = _pad_to_multiple(x, block_len, dim, pad_value=0)
    num_blocks = x.shape[dim] // block_len
    output_shape = x.shape[:dim] + (num_blocks, block_len) + x.shape[(dim + 1) :]
    # If 0 is in output_shape, we cannot apply reshape because of incompatibility with ONNX conversion
    if 0 in output_shape:
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)
    return x.reshape(output_shape)


def _concatenate_3_blocks(x: torch.Tensor, block_dim: int, sequence_dim: int, pad_value: int = 0) -> torch.Tensor:
    """Concatenate three consecutive blocks for each input block for local attentiont.

    For more information, see: https://arxiv.org/pdf/2112.07916.pdf.
    """
    num_blocks = x.shape[block_dim]

    pad = [(0, 0)] * x.ndim
    pad[block_dim] = (1, 1)
    pad = sum(pad[::-1], ())
    # [batch_size, num_blocks, block_len] -> [batch_size, num_blocks + 2, block_len]
    x = F.pad(x, pad=pad, mode="constant", value=pad_value)

    blocks_list: List[torch.Tensor] = []
    for i in range(3):
        # We use indexing approach here:
        # https://numpy.org/doc/stable/user/basics.indexing.html#dealing-with-variable-numbers-of-indices-within-programs
        indices = [slice(0, None)] * x.ndim
        indices[block_dim] = slice(i, i + num_blocks)
        indices = tuple(indices)
        blocks_list.append(x[indices])
    # [batch_size, num_blocks, 3 * block_len, ...]
    return torch.cat(blocks_list, dim=sequence_dim)


def make_3block_relative_position_ids(block_len: int) -> torch.Tensor:
    """Makes 3-blocked relative position ids for local attention."""
    position_ids = torch.arange(3 * block_len, dtype=torch.int32)
    center_position_ids = position_ids[block_len:-block_len]
    # [block_len, 3 * block_len]
    relative_position_ids = position_ids.unsqueeze(0) - center_position_ids.unsqueeze(1)
    return relative_position_ids


def _mask_local_attention_mask(local_attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Mask local attention mask to enforce that tokens are not allowed to attend tokens farther than ``local_radius."""
    relative_position_ids = make_3block_relative_position_ids(block_len)
    locality_mask = torch.abs(relative_position_ids) < block_len
    locality_mask = locality_mask[None, None, :, :]
    locality_mask = locality_mask.to(local_attention_mask.device)
    return torch.logical_and(local_attention_mask, locality_mask)


def get_local_attention_mask(attention_mask: torch.Tensor, block_len: int) -> torch.Tensor:
    """Prepare attention mask to be applied for a local attention."""
    # [batch_size, num_blocks, block_len]
    _blocked_attention_mask = _split_into_blocks(attention_mask, block_len, dim=1)
    # [batch_size, num_block, 3 * block_len]
    _3blocked_attention_mask = _concatenate_3_blocks(_blocked_attention_mask, block_dim=1, sequence_dim=2)

    _blocked_attention_mask = _blocked_attention_mask.unsqueeze(-1)
    _3blocked_attention_mask = _3blocked_attention_mask.unsqueeze(-2)
    # [batch_size, num_block, block_len, 3 * block_len]
    local_attention_mask = torch.logical_and(_blocked_attention_mask, _3blocked_attention_mask)
    local_attention_mask = _mask_local_attention_mask(local_attention_mask, block_len)

    # [batch_size, 1, num_block, block_len, 3 * block_len]
    return local_attention_mask.unsqueeze(1)


@with_incremental_state
class LocalAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        is_decoder=False,
        dropout=0.0,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        kwargs=None,
    ):
        super().__init__()

        # sanity check
        assert not encoder_decoder_attention, "local attention does not support cross attention"

        self.is_decoder = is_decoder
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
                self.num_heads * self.head_dim == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.inner_dim = self.num_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.has_relative_attention_bias = kwargs["has_relative_attention_bias"]
        self.relative_attention_num_buckets = kwargs["relative_attention_num_buckets"]
        self.relative_attention_max_distance = kwargs["relative_attention_max_distance"]
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

        self.local_radius = kwargs["local_radius"]
        self.block_len = self.local_radius + 1

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = quant_noise(
            nn.Linear(self.embed_dim, self.inner_dim, bias=False), q_noise, qn_block_size
        )
        self.k = quant_noise(
            nn.Linear(self.embed_dim, self.inner_dim, bias=False), q_noise, qn_block_size
        )
        self.v = quant_noise(
            nn.Linear(self.embed_dim, self.inner_dim, bias=False), q_noise, qn_block_size
        )
        self.o = quant_noise(
            nn.Linear(self.inner_dim, self.embed_dim, bias=False), q_noise, qn_block_size
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Empirically observed the convergence to be much better with
        # the scaled initialization
        nn.init.xavier_uniform_(self.k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q.weight, gain=1 / math.sqrt(2))

        nn.init.xavier_uniform_(self.o.weight)
        if self.o.bias is not None:
            nn.init.constant_(self.o.bias, 0.0)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, block_length: int):
        """Compute binned relative position bias"""
        memory_position = torch.arange(
            3 * block_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )
        context_position = memory_position[block_length:-block_length]

        # (block_length, 3 * block_length)
        relative_position = memory_position[None, :] - context_position[:, None]
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # (block_length, 3 * block_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # (block_length, 3 * block_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # (1, 1, num_heads, block_length, 3 * block_length)
        values = values.permute([2, 0, 1]).unsqueeze(0).unsqueeze(0)
        return values

    def forward(
        self,
        hidden_states,
        local_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Input shape: Time x Batch x Channel"""

        # Transformers receives input shape (Batch x Time x Channel)
        hidden_states = hidden_states.permute(1, 0, 2)

        batch_size, seq_length = hidden_states.shape[:2]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.num_heads, self.head_dim)

        def unshape(states):
            """reshape"""
            return states.contiguous().view(batch_size, -1, self.inner_dim)

        # get query/key/value states -> (batch_size, seq_length, n_heads, dim_per_head)
        query_states = shape(self.q(hidden_states))
        key_states = shape(self.k(hidden_states))
        value_states = shape(self.v(hidden_states))
        query_states *= self.scaling

        # Split into blocks -> (batch_size, num_blocks, block_len, n_heads, dim_per_head)
        query_states = _split_into_blocks(query_states, self.block_len, dim=1)
        key_states = _split_into_blocks(key_states, self.block_len, dim=1)
        value_states = _split_into_blocks(value_states, self.block_len, dim=1)

        # Concatenate 3 blocks for keys and values -> (batch_size, num_blocks, 3 * block_len, n_heads, dim_per_head)
        key_states = _concatenate_3_blocks(key_states, block_dim=1, sequence_dim=2)
        value_states = _concatenate_3_blocks(value_states, block_dim=1, sequence_dim=2)

        # Compute scores
        scores = torch.einsum(
            "...qhd,...khd->...hqk", query_states, key_states
        )  # (batch_size, num_block, n_heads, block_len, 3 * block_len)

        if position_bias is None or len(position_bias.size()) != 5:
            # position_bias shape: # (1, 1, n_heads, block_len, 3 * block_len)
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, 1, self.num_heads, self.block_len, 3 * self.block_len),
                    device=scores.device, dtype=scores.dtype
                )
            else:
                position_bias = self.compute_bias(self.block_len)

            if local_mask is not None:
                # Replace masked positions with -1e10 (according to the original implementation)
                local_mask = torch.where(local_mask > 0, 0.0, -1e10)
                # We need to adjust position bias shape to be sum with mask
                position_bias = position_bias + local_mask.transpose(1, 2)

        scores += position_bias
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        # (batch_size, num_blocks, n_heads, block_len, 3 * block_len)
        attn_weights = self.dropout_module(attn_weights)

        attn_weights = attn_weights.type(value_states.dtype)
        attn_output = unshape(torch.einsum("...hqk,...khd->...qhd", attn_weights, value_states))
        attn_output = attn_output[:, :seq_length, :]
        attn_output = self.o(attn_output)

        attn = attn_output.permute(1, 0, 2)
        return attn, None, position_bias

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result
