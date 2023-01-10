from dataclasses import dataclass, field
from typing import Optional

from fairseq import options
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveInput, CharacterTokenEmbedder
from fairseq.utils import safe_getattr

from fairseq.models.transformer_lm import (
    TransformerLanguageModelConfig,
    TransformerLanguageModel,
    base_lm_architecture,
    transformer_lm_big,
    transformer_lm_baevski_wiki103,
    transformer_lm_baevski_gbw
)

from .spade_model import (
    SpadeConfig,
    SpadeDecoderBase
)

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class SpadeLanguageModelConfig(TransformerLanguageModelConfig):
    # TODO: this is the same as SpadeConfig, need to fix this
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


@register_model("spade_lm", dataclass=SpadeLanguageModelConfig)
class SpadeLanguageModel(TransformerLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if safe_getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = safe_getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary,
                eval(args.character_filters),
                args.character_embedding_dim,
                args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary),
                task.source_dictionary.pad(),
                args.decoder_input_dim,
                args.adaptive_input_factor,
                args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(
                args, task.source_dictionary, args.decoder_input_dim
            )

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert (
                    args.adaptive_softmax_cutoff == args.adaptive_input_cutoff
            ), "{} != {}".format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff
            )
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = SpadeDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)


class SpadeDecoder(SpadeDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        cfg = SpadeConfig.from_namespace(args)
        super().__init__(
            cfg,
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            SpadeConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            SpadeConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )


def spade_base_lm_architecture(args):
    base_lm_architecture(args)


@register_model_architecture("spade_lm", "spade_lm_big")
def spade_lm_big(args):
    transformer_lm_big(args)


@register_model_architecture("spade_lm", "spade_lm_wiki103")
@register_model_architecture("spade_lm", "spade_lm_baevski_wiki103")
def spade_lm_baevski_wiki103(args):
    transformer_lm_baevski_wiki103(args)


@register_model_architecture("spade_lm", "spade_lm_gbw")
@register_model_architecture("spade_lm", "spade_lm_baevski_gbw")
def spade_lm_baevski_gbw(args):
    transformer_lm_baevski_gbw(args)
