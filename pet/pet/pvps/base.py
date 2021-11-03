import string
from abc import ABC, abstractmethod
from typing import Tuple, List, Union, Dict
import log
import torch

logger = log.get_logger("root")


class Verbalizer(ABC):
    def __init__(self):
        raise ValueError

    def verbalize(self, label):
        pass


class AtomPVP(ABC):
    """atomic PVP contains all possible pattern_id and language combos"""

    def __init__(self, wrapper, pattern_id: int):
        self.wrapper = wrapper  # wrapped transformer model
        self.pattern_id = pattern_id

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @staticmethod
    def shortenable(s):
        """mark this string as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return AtomPVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return AtomPVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(
            len(self.verbalize(label)) for label in self.wrapper.config.label_list
        )
