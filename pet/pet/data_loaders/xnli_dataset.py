import json
import os
from typing import List
from .base import DataProcessor, InfoEgs
from pet.utils import InputExample
from collections import defaultdict


class XNLIDataset(DataProcessor):
    def __init__(self, num_shots, data_lang=None):
        super().__init__()
        self.langs = [
            "en",
            "fr",
            "de",
            "es",
            "ru",
            "zh",
            "ar",
            "bg",
            "el",
            "hi",
            "sw",
            "th",
            "tr",
            "ur",
            "vi",
        ]
        self.data_lang = data_lang
        self.num_shots = num_shots
        if data_lang:
            self.langs = [data_lang]
            self.trn_egs = InfoEgs(self.get_train_examples(), "trn", data_lang)
            self.val_egs = InfoEgs(self.get_dev_examples(), "val", data_lang)
        else:
            self.trn_egs = InfoEgs(self.get_train_examples(), "trn", "en")
            self.val_egs = InfoEgs(self.get_dev_examples(), "val", "en")
        self.zs_egs = {}
        for lang, egs in self.get_transfer_examples().items():
            self.zs_egs[lang] = InfoEgs(egs, "tst", lang)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    def get_train_examples(self):
        if not self.data_lang:
            trn_data_path = "common_datasets/train"
            return self._create_examples(
                os.path.join(
                    trn_data_path, "en", f"train_split,{self.num_shots}.jsonl"
                ),
                "trn",
                "en",
            )
        else:
            trn_data_path = "common_datasets/train"
            return self._create_examples(
                os.path.join(
                    trn_data_path,
                    "rest",
                    f"train_split,{self.data_lang},{self.num_shots}.jsonl",
                ),
                "trn",
                self.data_lang,
            )

    def get_dev_examples(self):
        if not self.data_lang:
            dev_data_path = "common_datasets/dev"
            return self._create_examples(
                os.path.join(dev_data_path, "en", f"dev_split,{self.num_shots}.jsonl"),
                "val",
                "en",
            )
        else:
            dev_data_path = "common_datasets/dev"
            return self._create_examples(
                os.path.join(
                    dev_data_path,
                    "rest",
                    f"dev_split,{self.data_lang},{self.num_shots}.jsonl",
                ),
                "val",
                self.data_lang,
            )

    def get_transfer_examples(self):
        test_data_ = "common_datasets/test/xnli.test.jsonl"
        lang2egs = defaultdict(list)
        with open(test_data_, "r") as f:
            for idx, line in enumerate(f):
                line = json.loads(line.strip())
                lang = line["language"]
                label = line["gold_label"]
                text_a = line["sentence1"]
                text_b = line["sentence2"]
                guid = f"{lang}-{idx}-zstransfer"
                example = InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label, lang=lang
                )
                lang2egs[lang].append(example)
        for lang, egs in lang2egs.items():
            assert len(egs) == 5010
        return {l: v for l, v in lang2egs.items() if l in self.langs}

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def _create_examples(
        self, path: str, set_type: str, lang: str
    ) -> List[InputExample]:
        examples = []
        with open(path) as f:
            for idx, row in enumerate(f):
                row = json.loads(row)
                if "language" in row:
                    language = row["language"]
                elif "lang" in row:
                    language = row["lang"]
                else:
                    raise ValueError
                assert language == lang
                text_a = row.get("sentence1")
                text_b = row.get("sentence2")
                label = row.get("gold")
                assert label in self.get_labels()
                guid = "%s-%s" % (set_type, idx)
                example = InputExample(
                    guid=guid, text_a=text_a, text_b=text_b, label=label, lang=lang
                )
                examples.append(example)
        return examples
