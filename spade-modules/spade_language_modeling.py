import logging
import os
import torch

from dataclasses import dataclass, field

from fairseq import utils
from fairseq.data import (
    MonolingualDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask

from .local_attention_monolingual_dataset import LocalAttentionMonolingualDataset

logger = logging.getLogger(__name__)


@dataclass
class SpadeLanguageModelingConfig(LanguageModelingConfig):
    attention_type: str = field(
        default="full",
        metadata={"help": "type of attention"},
    )
    local_radius: int = field(
        default=127,
        metadata={"help": "radius of local attention"},
    )


@register_task("spade_language_modeling", dataclass=SpadeLanguageModelingConfig)
class SpadeLanguageModelingTask(LanguageModelingTask):
    def __init__(self, args, dictionary, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary, targets)

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> MonolingualDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        # each process has its own copy of the raw data (likely to be an np.memmap)
        dataset = data_utils.load_indexed_dataset(
            split_path, self.dictionary, self.args.dataset_impl, combine=combine
        )
        if dataset is None:
            raise FileNotFoundError(f"Dataset not found: {split} ({split_path})")

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
            use_plasma_view=self.args.use_plasma_view,
            split_path=split_path,
            plasma_path=self.args.plasma_path,
        )

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        if self.args.attention_type == "local":
            self.datasets[split] = LocalAttentionMonolingualDataset(
                dataset=dataset,
                sizes=dataset.sizes,
                src_vocab=self.dictionary,
                tgt_vocab=self.output_dictionary,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=True,
                targets=self.targets,
                add_bos_token=self.args.add_bos_token,
                fixed_pad_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
                local_radius=self.args.local_radius,
            )
        else:
            self.datasets[split] = MonolingualDataset(
                dataset=dataset,
                sizes=dataset.sizes,
                src_vocab=self.dictionary,
                tgt_vocab=self.output_dictionary,
                add_eos_for_other_targets=add_eos_for_other_targets,
                shuffle=True,
                targets=self.targets,
                add_bos_token=self.args.add_bos_token,
                fixed_pad_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
