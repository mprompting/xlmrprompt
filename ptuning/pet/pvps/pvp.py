import string
import torch
from collections import defaultdict
from typing import Tuple, List, Union, Dict
from pet.utils import InputExample, get_verbalization_ids
from .base import AtomPVP, Verbalizer
from abc import abstractmethod
from transformers import PreTrainedTokenizer, GPT2Tokenizer
import log

logger = log.get_logger("root")

FilledPattern = Tuple[
    List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]
]


class LMPVP(AtomPVP):
    def __init__(self, wrapper, pattern_id: int):
        super(LMPVP, self).__init__(wrapper, pattern_id)
        self.mlm_logits_to_cls_logits_tensor = (
            self._build_mlm_logits_to_cls_logits_tensor()
        )

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = (
            torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long)
            * -1
        )
        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(
                    verbalizer, self.wrapper.tokenizer, force_single_token=True
                )
                assert (
                    verbalizer_id != self.wrapper.tokenizer.unk_token_id
                ), f"your input strange verbolizer {verbalizer} was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    def convert_mlm_logits_to_cls_logits(
        self, mlm_labels: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack(
            [self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits]
        )
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(
        self, logits: torch.Tensor
    ) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        filler_len = torch.tensor(
            [len(self.verbalize(label)) for label in self.wrapper.config.label_list],
            dtype=torch.float,
        )
        filler_len = filler_len.to(logits.device)
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits


class PVP(LMPVP):
    def __init__(self, wrapper, pattern_id: int):
        super(PVP, self).__init__(wrapper, pattern_id)

    def verbalize(self, label: str):
        pass

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        pass

    def encode(self, example: InputExample) -> Tuple[List[int], List[int]]:
        tokenizer = self.wrapper.tokenizer
        parts_a, parts_b, block_flag_a, block_flag_b = self.get_parts(example)
        kwargs = (
            {"add_prefix_space": True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        )
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [
            (tokenizer.encode(x, add_special_tokens=False, **kwargs), s)
            for x, s in parts_a
            if x
        ]
        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [
                (tokenizer.encode(x, add_special_tokens=False, **kwargs), s)
                for x, s in parts_b
                if x
            ]
        num_special = self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(
            parts_a,
            parts_b,
            max_length=self.wrapper.config.max_seq_length - num_special,
        )
        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        if parts_b:
            tokens_b = [token_id for part, _ in parts_b for token_id in part]
        else:
            tokens_b = []
        assert len(parts_a) == len(
            block_flag_a
        ), f"{parts_a} \n {block_flag_a}, {len(parts_a)} vs {len(block_flag_a)}. {example}"
        assert len(parts_b) == len(block_flag_b)
        block_flag_a = [
            flag for (part, _), flag in zip(parts_a, block_flag_a) for _ in part
        ]
        block_flag_b = [
            flag for (part, _), flag in zip(parts_b, block_flag_b) for _ in part
        ]
        assert len(tokens_a) == len(block_flag_a)
        assert len(tokens_b) == len(block_flag_b)
        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(
                tokens_a, tokens_b
            )
            block_flag = tokenizer.build_inputs_with_special_tokens(
                block_flag_a, block_flag_b
            )
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)
            block_flag = tokenizer.build_inputs_with_special_tokens(block_flag_a)
        block_flag = [item if item in [0, 1] else 0 for item in block_flag]
        assert len(input_ids) == len(block_flag)
        return (input_ids, token_type_ids, block_flag)

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        if not parts:
            return 0
        all_lens = []
        for x, shortenable in parts:
            if not only_shortenable or shortenable:
                all_lens.append(len(x))
        return sum(all_lens)

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(
            idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq
        )
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(
        self,
        parts_a: List[Tuple[str, bool]],
        parts_b: List[Tuple[str, bool]],
        max_length: int,
    ):
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length
        if num_tokens_to_remove <= 0:
            return parts_a, parts_b
        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(
                parts_b, only_shortenable=True
            ):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)
