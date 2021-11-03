from .bert_formatting import encode_example_to_feature
import json, pickle
import torch
import os


class SeqClsDataIter(object):
    def __init__(self, raw_dataset, model, tokenizer, max_seq_len):
        self.raw_dataset = raw_dataset
        self.trn_egs = self.wrap_iter(
            task=raw_dataset.name,
            egs=raw_dataset.trn_egs,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        self.val_egs = self.wrap_iter(
            task=raw_dataset.name,
            egs=raw_dataset.val_egs,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        if raw_dataset.tst_egs is not None:
            self.tst_egs = self.wrap_iter(
                task=raw_dataset.name,
                egs=raw_dataset.tst_egs,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )

    def wrap_iter(self, task, egs, tokenizer, max_seq_len):
        if egs is None or len(egs) == 0:
            fts = []
        else:
            fts = []
            for idx, eg in enumerate(egs):
                if idx == 0:
                    loud = True
                else:
                    loud = False
                fts.append(
                    encode_example_to_feature(
                        eg, tokenizer, max_seq_len, self.label_list, loud
                    )
                )
        return _SeqClsIter(fts)

    @property
    def name(self):
        return self.raw_dataset.name

    @property
    def label_list(self):
        return self.raw_dataset.label_list


class _SeqClsIter(torch.utils.data.Dataset):
    def __init__(self, fts):
        self.uides = [ft.uid for ft in fts]
        self.input_idses = torch.as_tensor(
            [ft.input_ids for ft in fts], dtype=torch.long
        )
        self.golds = torch.as_tensor([ft.gold for ft in fts], dtype=torch.long)
        self.attention_maskes = torch.as_tensor(
            [ft.attention_mask for ft in fts], dtype=torch.long
        )
        self.token_type_idses = torch.as_tensor(
            [ft.token_type_ids for ft in fts], dtype=torch.long
        )

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.uides[idx],
            self.input_idses[idx],
            self.golds[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )
