from collections import namedtuple
from .nets import (
    BertForSequenceClassification,
    BertTokenizer,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
)
from .to_device import seqcls_batch_to_device, _Seqcls_task_container

Classes = namedtuple("Classes", "seqcls seqtag multiplechoice tokenizer")

ptl2classes = {
    "bert": Classes(BertForSequenceClassification, None, None, BertTokenizer),
    "xlm-roberta": Classes(
        XLMRobertaForSequenceClassification, None, None, XLMRobertaTokenizer
    ),
}


# collocate fns
batch_container = namedtuple("batch_container", "batch_to_device task_container")
seqcls_to_device = batch_container(seqcls_batch_to_device, _Seqcls_task_container)
