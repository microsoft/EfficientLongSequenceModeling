import torch

from fairseq.data import MonolingualDataset, data_utils

from .local_attention import get_local_attention_mask


def collate(samples, local_radius, pad_idx, eos_idx, fixed_pad_length=None, pad_to_bsz=None):
    if len(samples) == 0:
        return {}

    def merge(key, is_list=False):
        if is_list:
            res = []
            for i in range(len(samples[0][key])):
                res.append(
                    data_utils.collate_tokens(
                        [s[key][i] for s in samples],
                        pad_idx,
                        eos_idx,
                        left_pad=False,
                        pad_to_length=fixed_pad_length,
                        pad_to_bsz=pad_to_bsz,
                    )
                )
            return res
        else:
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx,
                left_pad=False,
                pad_to_length=fixed_pad_length,
                pad_to_bsz=pad_to_bsz,
            )

    src_tokens = merge("source")
    if samples[0]["target"] is not None:
        is_target_list = isinstance(samples[0]["target"], list)
        target = merge("target", is_target_list)
    else:
        target = src_tokens

    local_attention_mask = get_local_attention_mask(
        src_tokens.ne(pad_idx), local_radius + 1
    )

    return {
        "id": torch.LongTensor([s["id"] for s in samples]),
        "nsentences": len(samples),
        "ntokens": sum(len(s["source"]) for s in samples),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": torch.LongTensor([s["source"].numel() for s in samples]),
            "tgt_local_attention_mask": local_attention_mask,
        },
        "target": target,
    }


class LocalAttentionMonolingualDataset(MonolingualDataset):
    """
    A wrapper around torch.utils.data.Dataset for monolingual data.

    Args:
        dataset (torch.utils.data.Dataset): dataset to wrap
        sizes (List[int]): sentence lengths
        vocab (~fairseq.data.Dictionary): vocabulary
        shuffle (bool, optional): shuffle the elements before batching
            (default: True).
    """

    def __init__(
        self,
        dataset,
        sizes,
        src_vocab,
        tgt_vocab=None,
        add_eos_for_other_targets=False,
        shuffle=False,
        targets=None,
        add_bos_token=False,
        fixed_pad_length=None,
        pad_to_bsz=None,
        src_lang_idx=None,
        tgt_lang_idx=None,
        local_radius=None,
    ):
        super().__init__(
            dataset, sizes, src_vocab, tgt_vocab, add_eos_for_other_targets, shuffle,
            targets, add_bos_token, fixed_pad_length, pad_to_bsz, src_lang_idx, tgt_lang_idx
        )
        self.local_radius = local_radius

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the right.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the right.
        """
        return collate(
            samples,
            self.local_radius,
            self.vocab.pad(),
            self.vocab.eos(),
            self.fixed_pad_length,
            self.pad_to_bsz,
        )
