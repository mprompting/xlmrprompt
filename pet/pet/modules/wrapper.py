import json
import jsonpickle
import os
import copy
from typing import List, Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from .configs import (
    transformers_version,
    WrapperConfig,
    MODEL_CLASSES,
    PREPROCESSORS,
    InputExample,
    PLM_WRAPPER,
    MLM_WRAPPER,
    get_linear_schedule_with_warmup,
    TRAIN_STEP_FUNCTIONS,
    EVALUATION_STEP_FUNCTIONS,
    CONFIG_NAME,
)
from pet import preprocessor
from pet.data_loaders import TASK_HELPERS
from pet.utils import InputFeatures, DictDataset
from pet.trainers.singleton_evalfns import singleton_evaluate
from pet.preprocessor import Preprocessor

import log

logger = log.get_logger("root")


class TransformerModelWrapper:
    def __init__(self, config: WrapperConfig, train_config: WrapperConfig):
        self.config = config
        self.ce_criterion = nn.CrossEntropyLoss()
        config_class = MODEL_CLASSES[self.config.model_type]["config"]
        tokenizer_class = MODEL_CLASSES[self.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[self.config.model_type][self.config.wrapper_type]
        model_config = config_class.from_pretrained(
            config.model_name_or_path,
            num_labels=len(config.label_list),
            finetuning_task=config.task_name,
            cache_dir=config.cache_dir if config.cache_dir else None,
            use_cache=False,
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            config.model_name_or_path,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )  # type: PreTrainedTokenizer

        if self.config.model_type == "gpt2":
            self.tokenizer.pad_token, self.tokenizer.mask_token = (
                self.tokenizer.eos_token,
                self.tokenizer.eos_token,
            )

        self.model = model_class.from_pretrained(
            config.model_name_or_path,
            config=model_config,
            cache_dir=config.cache_dir if config.cache_dir else None,
        )
        self.task_helper = (
            TASK_HELPERS[self.config.task_name](self)
            if self.config.task_name in TASK_HELPERS
            else None
        )
        self.preprocessor = PREPROCESSORS[self.config.wrapper_type](
            self,
            self.config.task_name,
            train_config.pattern_lang,
            self.config.pattern_id,
        )

    @classmethod
    def from_pretrained(cls, path: str) -> "TransformerModelWrapper":
        """Load a pretrained wrapper from a given path."""
        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)
        tokenizer_class = MODEL_CLASSES[wrapper.config.model_type]["tokenizer"]
        model_class = MODEL_CLASSES[wrapper.config.model_type][
            wrapper.config.wrapper_type
        ]
        wrapper.model = model_class.from_pretrained(path)
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)
        wrapper.task_helper = (
            TASK_HELPERS[wrapper.config.task_name](wrapper)
            if wrapper.config.task_name in TASK_HELPERS
            else None
        )
        return wrapper

    def save(self, path: str) -> None:
        """Save a pretrained wrapper."""
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)

    def _save_config(self, path: str) -> None:
        with open(os.path.join(path, CONFIG_NAME), "w") as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str) -> WrapperConfig:
        with open(os.path.join(path, CONFIG_NAME), "r") as f:
            return jsonpickle.decode(f.read())

    def train(
        self,
        rank_idx,
        data_iterator,
        eval_config,
        device,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps=0,
        max_grad_norm: float = 1,
        logging_steps: int = 50,
        per_gpu_unlabeled_batch_size: int = 8,
        lm_training: bool = False,
        use_logits: bool = False,
        alpha: float = 0.8,
        temperature: float = 1,
        max_steps=-1,
        ddp_config=None,
        **_,
    ):
        self.ce_criterion = self.ce_criterion.to(device)
        train_batch_size = data_iterator.trn_batch_size
        train_dataloader = data_iterator.trn_iter
        val_dataloader = data_iterator.val_iter
        t_total = (
            len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon
        )
        if n_gpu > 1 and ddp_config is None:
            self.model = torch.nn.DataParallel(self.model)

        step, global_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        self.model.train()
        best_score, best_model = -1, None
        for epoch_index in range(int(num_train_epochs)):
            if ddp_config:
                data_iterator.train_sampler.set_epoch(epoch_index)
            for _, batch in enumerate(train_dataloader):
                batch = {k: t.to(device) for k, t in batch.items() if k != "guids"}
                loss = TRAIN_STEP_FUNCTIONS[self.config.wrapper_type](self)(batch)
                if n_gpu > 1 and not ddp_config:
                    loss = loss.mean()
                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                    self.model.zero_grad()
                    global_step += 1
                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        # learning_rate_scalar = scheduler.get_lr()[0]
                        learning_rate_scalar = learning_rate
                        logs["learning_rate"] = learning_rate_scalar
                        logs["regular_trn_loss"] = loss_scalar
                        logging_loss = tr_loss
                        print(json.dumps({**logs, **{"step": global_step}}))
                if step % 4 == 0:
                    logger.info(
                        f"rank{rank_idx} loss: {loss.item():.4f}, batch: {step}, epoch: {epoch_index}"
                    )
                step += 1
            logger.info(
                f"rank {rank_idx} epoch {epoch_index} finished. start evaluating."
            )
            eval_results = self.eval(val_dataloader, device)
            eval_results, _ = singleton_evaluate(eval_results, eval_config)
            logger.info(f"{eval_results['scores']}")
            if eval_results["scores"]["acc"] > best_score:
                best_score = eval_results["scores"]["acc"]
                best_model = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if rank_idx == 0:
            return (
                global_step,
                (tr_loss / global_step if global_step > 0 else -1),
                best_score,
                best_model,
            )
        else:
            return (None, None, None, None)

    def eval(self, eval_dataloader, device, zs_infer=False) -> Dict:
        self.model.eval()
        preds, all_indices, out_label_ids = None, None, None
        all_guids = []
        for batch in tqdm(eval_dataloader):
            guids = batch["guids"]
            all_guids.extend(guids)
            if not zs_infer:
                batch = {k: t.to(device) for k, t in batch.items() if k != "guids"}
            else:
                batch = {k: t.cuda() for k, t in batch.items() if k != "guids"}
            labels = batch["labels"]
            indices = batch["idx"]
            with torch.no_grad():
                logits = EVALUATION_STEP_FUNCTIONS[self.config.wrapper_type](self)(
                    batch
                )
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                all_indices = indices.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )
                all_indices = np.append(
                    all_indices, indices.detach().cpu().numpy(), axis=0
                )
        self.model.train()
        return {
            "indices": all_indices,
            "logits": preds,
            "labels": out_label_ids,
            "guids": all_guids,
        }

    def generate_default_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        if self.config.model_type in ["bert", "xlnet"]:
            inputs["token_type_ids"] = batch["token_type_ids"]
        return inputs

    def mlm_train_step(
        self,
        labeled_batch: Dict[str, torch.Tensor],
        unlabeled_batch: Optional[Dict[str, torch.Tensor]] = None,
        lm_training: bool = False,
        alpha: float = 0,
        **_,
    ) -> torch.Tensor:

        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch["mlm_labels"], labeled_batch["labels"]
        outputs = self.model(**inputs)
        prediction_scores = self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
            mlm_labels, outputs[0]
        )
        loss = self.ce_criterion(
            prediction_scores.view(-1, len(self.config.label_list)), labels.view(-1)
        )
        return loss

    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.preprocessor.pvp.convert_mlm_logits_to_cls_logits(
            batch["mlm_labels"], outputs[0]
        )

    def _mask_tokens(self, input_ids):
        """Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original."""
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        masked_indices = torch.bernoulli(probability_matrix).bool()
        if [int(v) for v in transformers_version.split(".")][:3] >= [2, 4, 0]:
            ignore_value = -100
        else:
            ignore_value = -1
        labels[~masked_indices] = ignore_value
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]
        return input_ids, labels
