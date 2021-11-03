from .xnli_dataset import XNLIDataset
from pet import task_helpers

name2datasets = {
    "xnli": XNLIDataset,
    "rte": None,
    "mldoc": None,
    "argmine": None,
}  # type: Dict[str,Callable[[],DataProcessor]]


METRICS = {
    "cb": ["acc", "f1-macro"],
    "multirc": ["acc", "f1", "em"],
    "rte": ["acc"],
    "mldoc": ["acc"],
    "xnli": ["acc"],
}

DEFAULT_METRICS = ["acc"]

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
UNLABELED_SET = "unlabeled"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, UNLABELED_SET]

TASK_HELPERS = {
    "wsc": task_helpers.WscTaskHelper,
    "multirc": task_helpers.MultiRcTaskHelper,
    "copa": task_helpers.CopaTaskHelper,
    "record": task_helpers.RecordTaskHelper,
}
