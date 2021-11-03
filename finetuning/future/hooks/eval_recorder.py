from .base_hook import Hook
from copy import deepcopy
import torch
import pickle
import os
import json


class EvaluationRecorder(Hook):
    def __init__(self, where_, which_metric="accuracy"):
        super(EvaluationRecorder, self).__init__()
        os.makedirs(where_, exist_ok=True)
        self.where_ = where_
        self.init_state_where_ = os.path.join(where_, "init_state.pt")
        self.best_state_where_ = os.path.join(where_, "best_state.pt")
        self.last_state_where_ = os.path.join(where_, "last_state.pt")
        self.tst_zipped_preds_where_ = os.path.join(where_, "zipped_preds.json")
        self.test_scores_where_ = os.path.join(where_, "test_scores.json")
        self.which_metric = which_metric
        self.name = self.__class__.__name__
        self.best_score = -1
        self.best_step = -1
        self.best_epoch = -1
        self.best_state = {}
        self.best_all_scores = None

    def on_train_begin(self):
        if not self.conf.train_fast:
            torch.save(self.get_state_dict(), self.init_state_where_)
            self.log_fn(f"[INFO] {self.name} saved initial state.")

    def on_batch_end(self):
        pass

    def on_validation_end(self, eval_score, all_scores):
        if eval_score is None:
            return
        curr_score = eval_score
        if curr_score > self.best_score:
            self.log_fn(
                f"[INFO] Update {self.name}: best {self.which_metric}={self.best_score:.5f} "
                f"@ {self.best_step} < {curr_score:.5f} (this batch @ {self.batch_step})."
            )
            self.best_step = deepcopy(self.batch_step)
            self.best_score = curr_score
            self.best_all_scores = all_scores
            self.best_epoch = deepcopy(self.epoch_step)
            self.best_state = {
                "which_step": self.best_step,
                "validation_score": self.best_score,
                "best_state_dict": deepcopy(self.model).cpu(),
                "best_all_scores": self.best_all_scores,
            }
        else:
            self.log_fn(
                f"[INFO] Not update {self.name}: best {self.which_metric}={self.best_score:.5f} "
                f"@ {self.best_step} > {curr_score:.5f} (this batch @ {self.batch_step})."
            )

    def on_train_end(self, tst_scores, tst_zipped_preds):
        self.best_state["tst_scores"] = tst_scores

        if self.where_ is None:
            self.log_fn(f"[INFO] {self.name} is NOT saving anything ... ")
            return
        if self.conf.train_fast:
            self.log_fn("[INFO] Fast training, zero out all state dicts.")
            self.best_state["best_state_dict"] = None
        else:
            self.log_fn("[INFO] Running ckpts.")
            self.best_state["best_state_dict"] = self.best_state[
                "best_state_dict"
            ].state_dict()
        torch.save(self.best_state, self.best_state_where_)
        with open(self.tst_zipped_preds_where_, "w") as f:
            json.dump(tst_zipped_preds, f)

        self.log_fn(
            f"[INFO] {self.name} saved validation results. "
            f" best score -> {self.best_score:.5f} @ {self.best_step}."
            f" best all scores -> {self.best_all_scores}"
            f" tst all scores -> {self.best_state['tst_scores']} "
        )
        with open(self.test_scores_where_, "w") as f:
            json.dump(tst_scores, f)

        if not self.conf.train_fast:
            torch.save(self.get_state_dict(), self.last_state_where_)
            self.log_fn(f"[INFO] {self.name} saved last state.")

    def get_state_dict(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module.state_dict()
        return self.model.state_dict()
