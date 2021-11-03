from torch.utils.data import SequentialSampler, RandomSampler
from .hooks import EvaluationRecorder
import utils.eval_meters as eval_meters
import torch


class BaseTrainer(object):
    def __init__(self, conf, logger, criterion=torch.nn.CrossEntropyLoss()):
        self.conf = conf
        self.logger = logger
        self.log_fn_json = logger.log_metric
        self.log_fn = logger.log
        self.criterion = criterion

        self._batch_step = 0
        self._epoch_step = 0

    @property
    def batch_step(self):
        return self._batch_step

    @property
    def epoch_step(self):
        return self._epoch_step

    def _parallel_to_device(self, model):
        model = model.cuda()
        if len(self.conf.world) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.conf.world)
        return model

    def _model_forward(self, model, **kwargs):
        if (
            self.model_ptl == "distilbert" or "roberta" in self.model_ptl
        ) and "token_type_ids" in kwargs:
            kwargs.pop("token_type_ids")
        return model(**kwargs)

    def _infer_one_loader(
        self, model, loader, collocate_batch_fn, metric_name="accuracy", device=None
    ):
        assert isinstance(loader.sampler, SequentialSampler)
        try:
            eval_fn = getattr(eval_meters, metric_name)
        except:
            raise ValueError(
                f"Required metric {metric_name} not implemented in meters module."
            )
        if device is None:
            device = torch.cuda.current_device()
        model.eval()
        all_golds, all_preds, all_uids = [], [], []
        for batched in loader:
            batched, golds, uids, *_ = collocate_batch_fn(batched, device=device)
            with torch.no_grad():
                logits, *_ = self._model_forward(model, **batched)
                preds = torch.argmax(logits, dim=-1)
            all_golds.extend(golds.tolist())
            all_preds.extend(preds.tolist())
            all_uids.extend(uids)
        assert len(all_golds) == len(all_preds) == len(all_uids)
        eval_meters.show_confusion(all_golds, all_preds)
        eval_res = eval_fn(all_golds, all_preds)
        model.train()
        return (eval_res, metric_name, list(zip(all_golds, all_preds, all_uids)))

    @staticmethod
    def _get_eval_recorder_hook(hook_container):
        for hook in hook_container.hooks:
            if isinstance(hook, EvaluationRecorder):
                return hook
