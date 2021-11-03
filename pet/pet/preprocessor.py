from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pet.utils import InputFeatures, InputExample
from pet.pvps.pvp import PVP
from pet.pvps import PVPS


class Preprocessor(ABC):
    def __init__(
        self,
        wrapper,
        task_name: str,
        pattern_lang: str,
        pattern_id: int = 0,
    ):
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_lang, pattern_id)
        self.label_map = {
            label: i for i, label in enumerate(self.wrapper.config.label_list)
        }

    @abstractmethod
    def get_input_features(
        self, example: InputExample, labelled: bool, priming: bool = False, **kwargs
    ) -> InputFeatures:
        pass


class MLMPreprocessor(Preprocessor):
    def get_input_features(self, example: InputExample, **kwargs) -> InputFeatures:

        input_ids, token_type_ids = self.pvp.encode(example)
        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)
        if padding_length < 0:
            raise ValueError(
                f"Maximum sequence length is too small, got {len(input_ids)} input ids"
            )

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert (
            len(token_type_ids) == self.wrapper.config.max_seq_length
        ), f"{len(token_type_ids)} {example}"
        label = self.label_map[example.label] if example.label is not None else -100

        # for ensembling
        logits = example.logits if example.logits else [-1]

        mlm_labels = self.pvp.get_mask_positions(input_ids)
        if self.wrapper.config.model_type == "gpt2":
            # shift labels to the left by one
            mlm_labels.append(mlm_labels.pop(0))

        return InputFeatures(
            guid=example.guid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label=label,
            mlm_labels=mlm_labels,
            logits=logits,
            idx=example.idx,
        )
