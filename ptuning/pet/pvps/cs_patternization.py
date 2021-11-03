from pet.utils import InputFeatures, DictDataset, InputExample
from pet.modules.configs import PLM_WRAPPER
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from typing import List, Dict
import torch
import numpy as np
import log

logger = log.get_logger("root")


class PatternizedIterator(object):
    def __init__(
        self,
        dataset,
        preprocessor,
        pattern_lang: str,
        trn_batch_size: int,
        inf_batch_size: int,
    ):
        self.preprocessor = preprocessor
        self.dataset = dataset
        self.pattern_lang = pattern_lang
        self.trn_batch_size = trn_batch_size
        self.inf_batch_size = inf_batch_size
        self.trn_iter = None
        self.val_iter = None
        self.zs_iters = None

    def patternize_trn(self):
        logger.info(f"Start patternizing training data (trn_egs)")
        logger.info(f"Traning data language: {self.dataset.trn_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.trn_egs.egs)
        self.trn_iter = self._wrap_sampler("trn", features)

    def patternize_val(self):
        logger.info(f"Start patternizing validation data (val_egs)")
        logger.info(f"Val data language: {self.dataset.val_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.val_egs.egs)
        self.val_iter = self._wrap_sampler("val", features)

    def patternize_zs(self):
        logger.info(f"Start patternizing zs transfer data (zs_egs)")
        logger.info(f"Pattern language: {self.pattern_lang}")
        self.zs_iters = {lang: None for lang, _ in self.dataset.zs_egs.items()}
        for lang, egs in self.dataset.zs_egs.items():
            logger.info(f"ZS data language: {egs.lang}")
            features = self._generate_dataset(egs)
            self.zs_iters[lang] = self._wrap_sampler("zs", features)

    def ddp_patternize_trn(self, ddp_config, rank_idx):
        logger.info(f"Refactoring trn_egs with DDP wrapper")
        logger.info(f"Traning data language: {self.dataset.trn_egs.lang}")
        logger.info(f"Pattern language: {self.pattern_lang}")
        features = self._generate_dataset(self.dataset.trn_egs.egs)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            features,
            num_replicas=ddp_config.world_size,
            rank=rank_idx,
            shuffle=True,
        )
        self.trn_iter = torch.utils.data.DataLoader(
            dataset=features,
            batch_size=self.trn_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=self.train_sampler,
        )

    def _wrap_sampler(self, split_, features):
        sampler = RandomSampler if split_ == "trn" else SequentialSampler
        bs = self.trn_batch_size if split_ == "trn" else self.inf_batch_size
        return DataLoader(features, sampler=sampler(features), batch_size=bs)

    def _convert_examples_to_features(
        self, examples: List[InputExample]
    ) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            input_features = self.preprocessor.get_input_features(example)
            features.append(input_features)
            if ex_index < 1:
                logger.info(f"--- Example {ex_index} ---")
                logger.info(
                    input_features.pretty_print(self.preprocessor.wrapper.tokenizer)
                )
        return features

    def _generate_dataset(self, data: List[InputExample]):
        features = self._convert_examples_to_features(data)
        feature_dict = {
            "input_ids": torch.tensor(
                [f.input_ids for f in features], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [f.attention_mask for f in features], dtype=torch.long
            ),
            "token_type_ids": torch.tensor(
                [f.token_type_ids for f in features], dtype=torch.long
            ),
            "labels": torch.tensor([f.label for f in features], dtype=torch.long),
            "mlm_labels": torch.tensor(
                [f.mlm_labels for f in features], dtype=torch.long
            ),
            "logits": torch.tensor([f.logits for f in features], dtype=torch.float),
            "idx": torch.tensor([f.idx for f in features], dtype=torch.long),
            "block_flag": torch.tensor(
                [f.block_flag for f in features], dtype=torch.long
            ),
            "guids": np.array([f.guid for f in features]),
        }
        return DictDataset(**feature_dict)
