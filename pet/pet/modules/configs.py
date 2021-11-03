from transformers import (
    InputExample,
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM,
    XLMRobertaForMaskedLM,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    XLNetLMHeadModel,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    AlbertForSequenceClassification,
    AlbertForMaskedLM,
    AlbertTokenizer,
    AlbertConfig,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers import __version__ as transformers_version
from pet import preprocessor
from typing import List


CONFIG_NAME = "wrapper_config.json"
SEQUENCE_CLASSIFIER_WRAPPER = "sequence_classifier"
MLM_WRAPPER = "mlm"
PLM_WRAPPER = "plm"

WRAPPER_TYPES = [SEQUENCE_CLASSIFIER_WRAPPER, MLM_WRAPPER, PLM_WRAPPER]

PREPROCESSORS = {
    SEQUENCE_CLASSIFIER_WRAPPER: None,
    MLM_WRAPPER: preprocessor.MLMPreprocessor,
    PLM_WRAPPER: None,
}

MODEL_CLASSES = {
    "bert": {
        "config": BertConfig,
        "tokenizer": BertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: BertForSequenceClassification,
        MLM_WRAPPER: BertForMaskedLM,
    },
    "roberta": {
        "config": RobertaConfig,
        "tokenizer": RobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: RobertaForSequenceClassification,
        MLM_WRAPPER: RobertaForMaskedLM,
    },
    "xlm-roberta": {
        "config": XLMRobertaConfig,
        "tokenizer": XLMRobertaTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLMRobertaForSequenceClassification,
        MLM_WRAPPER: XLMRobertaForMaskedLM,
    },
    "xlnet": {
        "config": XLNetConfig,
        "tokenizer": XLNetTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: XLNetForSequenceClassification,
        PLM_WRAPPER: XLNetLMHeadModel,
    },
    "albert": {
        "config": AlbertConfig,
        "tokenizer": AlbertTokenizer,
        SEQUENCE_CLASSIFIER_WRAPPER: AlbertForSequenceClassification,
        MLM_WRAPPER: AlbertForMaskedLM,
    },
    "gpt2": {
        "config": GPT2Config,
        "tokenizer": GPT2Tokenizer,
        MLM_WRAPPER: GPT2LMHeadModel,
    },
}

EVALUATION_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_eval_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_eval_step,
    SEQUENCE_CLASSIFIER_WRAPPER: lambda wrapper: None,
}

TRAIN_STEP_FUNCTIONS = {
    MLM_WRAPPER: lambda wrapper: wrapper.mlm_train_step,
    PLM_WRAPPER: lambda wrapper: wrapper.plm_train_step,
    SEQUENCE_CLASSIFIER_WRAPPER: None,
}


class WrapperConfig(object):
    def __init__(
        self,
        model_type: str,
        model_name_or_path: str,
        wrapper_type: str,
        task_name: str,
        max_seq_length: int,
        label_list: List[str],
        pattern_id: int = 0,
        cache_dir: str = None,
    ):
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.wrapper_type = wrapper_type
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.pattern_id = pattern_id
        self.cache_dir = cache_dir
