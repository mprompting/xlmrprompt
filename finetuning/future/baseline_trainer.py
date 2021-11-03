# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from .base import BaseTrainer
from .hooks.base_hook import HookContainer
from .hooks import EvaluationRecorder
from torch.utils.data import RandomSampler
from collections import defaultdict


class BaselineTuner(BaseTrainer):
    def __init__(self, conf, collocate_batch_fn, logger):
        assert len(conf.trn_languages) == 1
        super(BaselineTuner, self).__init__(conf, logger)
        self.log_fn("Init trainer.")
        self.collocate_batch_fn = collocate_batch_fn
        self.model_ptl = conf.ptl

    def _init_model_opt(self, model):
        model = self._parallel_to_device(model)
        opt = torch.optim.Adam(model.parameters(), lr=self.conf.finetune_lr)
        opt.zero_grad()
        model.zero_grad()
        return opt, model

    def train(self, model, tokenizer, data_iter, metric_name, hooks=None):
        opt, model = self._init_model_opt(model)
        self.model = model
        self.model.train()

        hook_container = HookContainer(world_env={"trainer": self}, hooks=hooks)
        hook_container.on_train_begin()

        assert len(self.conf.trn_languages) == 1
        train_iter = data_iter[self.conf.trn_languages[0]].trn_egs

        for epoch_index in range(1, self.conf.finetune_epochs + 1):
            epoch_losses = []
            for batch_index, batched in enumerate(train_iter):
                batched, golds, uids, _ = self.collocate_batch_fn(batched)
                logits, *_ = self._model_forward(self.model, **batched)
                loss = self.criterion(logits, golds).mean()
                epoch_losses.append(loss.item())
                loss.backward()
                opt.step()
                opt.zero_grad()
                self._batch_step += 1
                self.log_fn(
                    f"Traning loss on {self.conf.trn_languages[0]}: {loss.item():.3f}"
                    f" train batch  @  {batch_index}, epoch @ {epoch_index}"
                    f" global batch @ {self._batch_step}"
                )
                hook_container.on_batch_end()
            self._epoch_step += 1
            # now just eval after each epoch
            eval_score, all_scores = self.plain_eval(
                self.model, data_iter, metric_name=metric_name,
            )
            self.log_fn("--" * 10)
            self.log_fn(f"Evaluate @ epoch {epoch_index}:")
            self.log_fn(f"metrics: {metric_name}")
            self.log_fn(f"val score: {eval_score}, all: {all_scores.items()}")
            self.log_fn("--" * 10)
            hook_container.on_validation_end(
                eval_score=eval_score, all_scores=all_scores
            )
        tst_scores, all_zipped_preds = self._infer_tst_egs(
            hook_container, metric_name, data_iter, self.conf.eval_languages,
        )
        hook_container.on_train_end(
            learning_curves=None,
            tst_scores=tst_scores,
            tst_zipped_preds=all_zipped_preds,
        )
        return

    def _infer_tst_egs(self, hook_container, metric_name, data_iter, tst_languages):
        assert isinstance(tst_languages, list)
        best_model = deepcopy(
            self._get_eval_recorder_hook(hook_container).best_state["best_state_dict"]
        ).cuda()
        scores = defaultdict(dict)
        all_zipped_preds = defaultdict(dict)
        split_name = "tst_egs"
        for language in tst_languages:
            loader = getattr(data_iter[language], split_name)
            eval_res, _, zipped_preds = self._infer_one_loader(
                best_model, loader, self.collocate_batch_fn, metric_name=metric_name,
            )
            self.log_fn(f"{language} {split_name} score: {eval_res * 100:.1f}")
            scores[language][split_name] = eval_res
            all_zipped_preds[language][split_name] = zipped_preds
        return (scores, all_zipped_preds)

    def plain_eval(self, model, data_iter, metric_name):
        all_scores = defaultdict(list)
        val_scores = []
        for val_language in self.conf.trn_languages:
            val_loaders = data_iter[val_language]
            for split_ in ("val_egs", "tst_egs"):
                val_loader = getattr(val_loaders, split_)
                eval_res, *_ = self._infer_one_loader(
                    model, val_loader, self.collocate_batch_fn, metric_name=metric_name
                )
                all_scores[val_language].append((split_, eval_res))
                if split_ == "val_egs":
                    val_scores.append(eval_res)
        assert len(val_scores) == len(self.conf.trn_languages)
        return (np.mean(val_scores), all_scores)

