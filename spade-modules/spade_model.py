import torch

from dataclasses import dataclass, field
from torch import Tensor
from typing import Any, Dict, List, Optional

from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer.transformer_config import TransformerConfig
from fairseq.models.transformer.transformer_decoder import TransformerDecoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper

from .local_attention import get_local_attention_mask, make_3block_relative_position_ids
from .spade_layer import SpadeDecoderLayerBase


@dataclass
class SpadeConfig(TransformerConfig):
    attention_type: str = field(
        default="full",
        metadata={"help": "type of attention"},
    )

    # relative attention parameters
    use_relative_attention: bool = field(
        default=False,
        metadata={"help": "whether to use relative attention"},
    )
    relative_attention_num_buckets: int = field(
        default=32,
        metadata={"help": "radius of local attention"},
    )
    relative_attention_max_distance: int = field(
        default=128,
        metadata={"help": "radius of local attention"},
    )

    # local attention parameters
    local_radius: int = field(
        default=127,
        metadata={"help": "radius of local attention"},
    )
    s4_every_n_layers: int = field(
        default=1,
        metadata={"help": "use S4 every N layers"},
    )
    s4_local_combine: str = field(
        default="add",
        metadata={"help": "whether to concat/add/stack S4 and local attention"},
    )
    s4_weight: float = field(
        default=0.5,
        metadata={"help": "weight (between 0.0 and 1.0) of S4 when adding with local attention"},
    )

    # S4 parameters
    s4_state_dim: int = field(
        default=64,
        metadata={"help": "state dimension"},
    )
    s4_channels: int = field(
        default=1,
        metadata={"help": "number of channels (heads), default to 1"},
    )
    s4_dt_min: float = field(
        default=0.001,
        metadata={"help": "parameter for time steps"},
    )
    s4_dt_max: float = field(
        default=0.1,
        metadata={"help": "parameter for time steps"},
    )
    s4_lr: Optional[str] = field(
        default=None,
        metadata={"help": "learning rate for the state space parameters, except dt"},
    )
    s4_n_ssm: int = field(
        default=1,
        metadata={"help": "copies of the state space parameters A and B"},
    )


class SpadeDecoderBase(TransformerDecoderBase):
    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.layer_idx = 0
        super().__init__(cfg, dictionary, embed_tokens, no_encoder_attn, output_projection)

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        use_relative_attention = cfg.use_relative_attention and bool(self.layer_idx == 0)

        if cfg.attention_type == "local":
            layer = SpadeDecoderLayerBase(
                cfg, self.layer_idx, no_encoder_attn,
                has_relative_attention_bias=use_relative_attention,
            )
        else:
            raise NotImplementedError("attention type not implemented")

        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        self.layer_idx += 1
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        src_local_attention_mask: Optional[Tensor] = None,
        tgt_local_attention_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            local_attention_mask=tgt_local_attention_mask,
        )

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        local_attention_mask: Optional[Tensor] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            local_attention_mask,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        local_attention_mask: Optional[Tensor] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        use_local_attention = self.cfg.attention_type == "local"
        if use_local_attention:
            # add causal masking
            if local_attention_mask is None and not self.training:
                # this should only be used during inference
                local_attention_mask = get_local_attention_mask(
                    prev_output_tokens.ne(self.padding_idx), self.cfg.local_radius + 1
                ).to(prev_output_tokens.device)
            assert local_attention_mask is not None
            relative_position_ids = make_3block_relative_position_ids(self.cfg.local_radius + 1)
            causal_attention_mask = (relative_position_ids <= 0).to(local_attention_mask.device)
            local_attention_mask = torch.logical_and(local_attention_mask, causal_attention_mask)

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        position_bias: Optional[Tensor] = None
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            if use_local_attention:
                x, layer_attn, _, position_bias = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                    local_attn_mask=local_attention_mask,
                    position_bias=position_bias,
                )
            else:
                x, layer_attn, _ = layer(
                    x,
                    enc,
                    padding_mask,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}
