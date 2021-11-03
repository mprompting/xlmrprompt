from ..common import (
    SentencePairExample,
    MultilingualRawDataset,
    RawDataset,
)
import itertools
import os
import json
from collections import OrderedDict
from ..data_configs import abbre2language


class XNLIDataset(MultilingualRawDataset):
    def __init__(self, num_shots, in_lang=None):
        assert num_shots > 0
        self.name = "xnli"
        self.in_lang = in_lang
        self.lang_abbres = [
            "ar",
            "bg",
            "de",
            "el",
            "en",
            "es",
            "fr",
            "hi",
            "ru",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
            "zh",
        ]
        self.metrics = ["accuracy"]
        self.label_list = ["contradiction", "entailment", "neutral"]
        self.label2idx = {"contradiction": 0, "entailment": 1, "neutral": 2}
        self.num_labels = 3
        self.num_shots = num_shots
        self.contents = OrderedDict()
        self.create_contents()

    def get_labels(self):
        return self.label_list

    def get_language_data(self, language):
        return self.contents[language]

    def create_contents(self):
        entries = []
        if not self.in_lang:
            mnli_ = "common_datasets/train/en"
            file_ = os.path.join(mnli_, f"train_split,{self.num_shots}.jsonl")
            entries.extend(self.mnli_parse(file_, "trn"))
        else:
            mnli_ = "common_datasets/train/rest"
            file_ = os.path.join(
                mnli_, f"train_split,{self.in_lang},{self.num_shots}.jsonl"
            )
            entries.extend(self.mnli_parse(file_, "trn"))

        if not self.in_lang:
            mnli_ = "common_datasets/dev/en"
            file_ = os.path.join(mnli_, f"dev_split,{self.num_shots}.jsonl")
            entries.extend(self.mnli_parse(file_, "val"))
        else:
            mnli_ = "common_datasets/dev/rest"
            file_ = os.path.join(
                mnli_, f"dev_split,{self.in_lang},{self.num_shots}.jsonl"
            )
            entries.extend(self.mnli_parse(file_, "val"))

        xnli_ = "common_datasets/test"
        file_ = os.path.join(xnli_, "xnli.test.jsonl")
        entries.extend(self.xnli_parse(file_, "tst"))

        entries = sorted(entries, key=lambda x: x[0])
        for language, triplets in itertools.groupby(entries, key=lambda x: x[0]):
            triplets = list(triplets)
            trn_egs, val_egs, tst_egs = [], [], []
            for _, split, eg in triplets:
                if split == "trn":
                    trn_egs.append(eg)
                elif split == "val":
                    val_egs.append(eg)
                elif split == "tst":
                    tst_egs.append(eg)
                else:
                    raise ValueError
            _dataset = RawDataset(
                name=f"{self.name}-{language}",
                language=language,
                metrics=self.metrics,
                label_list=self.label_list,
                label2idx=self.label2idx,
            )
            _dataset.trn_egs = trn_egs if len(trn_egs) else None
            _dataset.val_egs = val_egs if len(val_egs) else None
            _dataset.tst_egs = tst_egs if len(tst_egs) else None
            self.contents[language] = _dataset

    def mnli_parse(self, input_file, which_split):
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                lang = line["lang"]
                text_a = line.get("sentence1").strip()
                text_b = line.get("sentence2").strip()
                label = line.get("gold")
                assert label in self.get_labels(), f"{label}, {input_file}"
                sentence_pair_egs.append(
                    (
                        abbre2language[lang],
                        which_split,
                        SentencePairExample(
                            uid=f"{abbre2language[lang]}-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                        ),
                    )
                )
        return sentence_pair_egs

    def xnli_parse(self, input_file, which_split):
        sentence_pair_egs = []
        with open(input_file, "r") as f:
            for idx, line in enumerate(f):
                line = json.loads(line.strip())
                lang = line["language"]
                label = line["gold_label"]
                text_a = line["sentence1"]
                text_b = line["sentence2"]
                sentence_pair_egs.append(
                    (
                        abbre2language[lang],
                        which_split,
                        SentencePairExample(
                            uid=f"{abbre2language[lang]}-{idx}-{which_split}",
                            text_a=text_a,
                            text_b=text_b,
                            label=label,
                        ),
                    ),
                )
        print(len(sentence_pair_egs), input_file)
        return sentence_pair_egs
