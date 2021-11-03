import copy
import json
import pickle
import random
import string
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer


def find_free_port():
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(
        self,
        guid,
        text_a,
        text_b=None,
        label=None,
        lang=None,
        logits=None,
        meta: Optional[Dict] = None,
        idx=-1,
    ):
        self.lang = lang
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List["InputExample"]:
        """Load a set of input examples from a file"""
        with open(path, "rb") as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List["InputExample"], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, "wb") as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """A set of numeric features obtained from an :class:`InputExample`"""

    def __init__(
        self,
        guid,
        input_ids,
        attention_mask,
        token_type_ids,
        label,
        mlm_labels=None,
        logits=None,
        meta: Optional[Dict] = None,
        idx=-1,
    ):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.mlm_labels = mlm_labels
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def pretty_print(self, tokenizer):
        return (
            f"input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n"
            + f"attention_mask    = {self.attention_mask}\n"
            + f"token_type_ids    = {self.token_type_ids}\n"
            + f"mlm_labels        = {self.mlm_labels}\n"
            + f"logits            = {self.logits}\n"
            + f"label             = {self.label}\n"
            + f"guid             = {self.guid}"
        )

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(
            next(iter(tensors.values())).shape[0] == tensor.shape[0]
            for tensor in tensors.values()
        )
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)


def set_seed(seed: int):
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eq_div(N, i):
    """Equally divide N examples among i buckets. For example, `eq_div(12,3) = [4,4,4]`."""
    return [] if i <= 0 else [N // i + 1] * (N % i) + [N // i] * (i - N % i)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def remove_final_punc(s: str):
    """Remove the last character from a string if it is some form of punctuation"""
    return s.rstrip(string.punctuation)


def lowercase_first(s: str):
    """Lowercase the first letter of a string"""
    return s[0].lower() + s[1:]


def save_logits(path: str, logits: np.ndarray):
    """Save an array of logits to a file"""
    with open(path, "w") as fh:
        for example_logits in logits:
            fh.write(" ".join(str(logit) for logit in example_logits) + "\n")


def save_predictions(path: str, wrapper, results: Dict):
    """Save a sequence of predictions to a file"""
    predictions_with_idx = []

    if wrapper.task_helper and wrapper.task_helper.output:
        predictions_with_idx = wrapper.task_helper.output
    else:
        inv_label_map = {
            idx: label for label, idx in wrapper.preprocessor.label_map.items()
        }
        for idx, prediction_idx in zip(results["indices"], results["predictions"]):
            prediction = inv_label_map[prediction_idx]
            idx = idx.tolist() if isinstance(idx, np.ndarray) else int(idx)
            predictions_with_idx.append({"idx": idx, "label": prediction})

    with open(path, "w", encoding="utf8") as fh:
        for line in predictions_with_idx:
            fh.write(json.dumps(line) + "\n")


def softmax(x, temperature=1.0, axis=None):
    """Custom softmax implementation"""
    y = np.atleast_2d(x)

    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    y = y * float(temperature)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum

    if len(x.shape) == 1:
        p = p.flatten()
    return p


def get_verbalization_ids(
    word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool
) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the v
           erbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a
           list and throws an error if the word corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {"add_prefix_space": True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    if len(ids) == 1:
        assert (
            len(ids) == 1
        ), f'Verbalization "{word}" does not correspond to a single token, `{word}` becomes {tokenizer.convert_ids_to_tokens(ids)}'
        verbalization_id = ids[0]
    else:
        print("*" * 20)
        print("Make sure you are training on Chinese/Thai")
        verbalization_id = ids[-1]
        print(verbalization_id, word, ids)
        print("*" * 20)
    assert (
        verbalization_id not in tokenizer.all_special_ids
    ), f"Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}"
    return verbalization_id


def trim_input_ids(
    input_ids: torch.tensor, pad_token_id, mask_token_id, num_masks: int
):
    """
    Trim a sequence of input ids by removing all padding tokens and keeping at most a specific number of mask tokens.

    :param input_ids: the sequence of input token ids
    :param pad_token_id: the id of the pad token
    :param mask_token_id: the id of the mask tokens
    :param num_masks: the number of masks to keeps
    :return: the trimmed sequence of input ids
    """
    assert input_ids.shape[0] == 1
    input_ids_without_pad = [x for x in input_ids[0] if x != pad_token_id]

    trimmed_input_ids = []
    mask_count = 0
    for input_id in input_ids_without_pad:
        if input_id == mask_token_id:
            if mask_count >= num_masks:
                continue
            mask_count += 1
        trimmed_input_ids.append(input_id)

    return torch.tensor([trimmed_input_ids], dtype=torch.long, device=input_ids.device)


def exact_match(predictions: np.ndarray, actuals: np.ndarray, question_ids: np.ndarray):
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em
