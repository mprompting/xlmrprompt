from abc import ABC
from typing import List
import json


class PetConfig(ABC):
    def __repr__(self):
        return repr(self.__dict__)

    def save(self, path: str):
        """Save this config to a file."""
        with open(path, "w", encoding="utf8") as fh:
            json.dump(self.__dict__, fh)

    @classmethod
    def load(cls, path: str):
        """Load a config from a file."""
        cfg = cls.__new__(cls)
        with open(path, "r", encoding="utf8") as fh:
            cfg.__dict__ = json.load(fh)
        return cfg


class TrainConfig(PetConfig):
    def __init__(
        self,
        pattern_lang: str,
        device: str = None,
        per_gpu_train_batch_size: int = 8,
        per_gpu_unlabeled_batch_size: int = 8,
        n_gpu: int = 1,
        num_train_epochs: int = 3,
        max_steps: int = -1,
        gradient_accumulation_steps: int = 1,
        weight_decay: float = 0.0,
        learning_rate: float = 5e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        max_grad_norm: float = 1,
        lm_training: bool = False,
        use_logits: bool = False,
        alpha: float = 0.9999,
        temperature: float = 1,
        seed: int = 0,
    ):
        self.pattern_lang = pattern_lang
        self.device = device
        self.per_gpu_train_batch_size = per_gpu_train_batch_size
        self.per_gpu_unlabeled_batch_size = per_gpu_unlabeled_batch_size
        self.n_gpu = n_gpu
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.lm_training = lm_training
        self.use_logits = use_logits
        self.alpha = alpha
        self.temperature = temperature
        self.seed = seed


class EvalConfig(PetConfig):
    def __init__(
        self,
        device: str = None,
        n_gpu: int = 1,
        per_gpu_eval_batch_size: int = 8,
        metrics: List[str] = None,
        decoding_strategy: str = "default",
    ):
        """
        :param decoding_strategy: the decoding strategy for PET with multiple masks ('default', 'ltr', or 'parallel')
        """
        self.device = device
        self.n_gpu = n_gpu
        self.per_gpu_eval_batch_size = per_gpu_eval_batch_size
        self.metrics = metrics
        self.decoding_strategy = decoding_strategy


class DDPConfig(PetConfig):
    def __init__(self, do_ddp, num_ranks, num_nodes, world_size=-1):
        super().__init__()
        self.do_ddp = do_ddp
        self.num_ranks = num_ranks
        self.num_nodes = num_nodes
        self.world_size = world_size


class IPetConfig(PetConfig):
    pass
