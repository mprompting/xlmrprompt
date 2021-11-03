import json
import copy
import torch
from tqdm import tqdm


class BertInputFeature(object):
    def __init__(self, uid, input_ids, attention_mask, token_type_ids, label):
        self.uid = uid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.gold = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def truncate(tokenizer, parts_a, parts_b, max_seq_len):
    len_parts_b = 0 if not parts_b else len(parts_b)
    total_len = len(parts_a) + len_parts_b
    total_len += tokenizer.num_special_tokens_to_add(bool(parts_b))
    num_tokens_to_remove = total_len - max_seq_len

    if num_tokens_to_remove <= 0:
        return (parts_a, parts_b)

    for _ in range(num_tokens_to_remove):
        if len(parts_a) > len_parts_b:
            parts_a = parts_a[:-1]
        elif parts_b:
            parts_b = parts_b[:-1]
    return (parts_a, parts_b)


def encode_example_to_feature(example, tokenizer, max_seq_len, label_list, loud=False):
    parts_a = tokenizer.encode(
        example.text_a, add_special_tokens=False, max_length=512, truncation=True
    )
    parts_b = None
    if hasattr(example, "text_b"):
        parts_b = tokenizer.encode(
            example.text_b, add_special_tokens=False, max_length=512, truncation=True
        )

    parts_a, parts_b = truncate(tokenizer, parts_a, parts_b, max_seq_len)
    input_ids = tokenizer.build_inputs_with_special_tokens(parts_a, parts_b)
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(parts_a, parts_b)
    attention_mask = [1] * len(input_ids)
    padding_length = max_seq_len - len(input_ids)
    if padding_length < 0:
        raise ValueError(
            f"Maximum sequence length is too small, got {len(input_ids)} input ids"
        )

    input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    if loud:
        print(example)
        print(parts_a)
        print(parts_b)
        print(input_ids)
        print(attention_mask)
        print(token_type_ids)
    assert len(input_ids) == max_seq_len
    assert len(attention_mask) == max_seq_len
    assert len(token_type_ids) == max_seq_len, f"{len(token_type_ids)} {example}"

    label_map = {l: i for i, l in enumerate(label_list)}

    return BertInputFeature(
        uid=example.uid,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        label=label_map[example.label],
    )
