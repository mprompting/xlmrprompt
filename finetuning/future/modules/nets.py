from transformers import (
    BertModel,
    BertPreTrainedModel,
    XLMRobertaForMaskedLM,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)
from transformers.modeling_bert import (
    BertForSequenceClassification,
    BertForMultipleChoice,
)
from transformers.configuration_roberta import RobertaConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_roberta import RobertaTokenizer

import torch.nn as nn
import torch


class LinearPredictor(BertPreTrainedModel):
    def __init__(self, bert_config, out_dim, dropout):
        super(LinearPredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def forward(self):
        raise NotImplementedError


class CosinePredictor(BertPreTrainedModel):
    """
    compute logits according to cosine similarity, should not use dropout
    """

    def __init__(self, bert_config, out_dim, scale_alpha=20, dropout=0.1):
        super(CosinePredictor, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.classifier = nn.Linear(768, out_dim)
        self.cos_linear = nn.Linear(768, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale_alpha = scale_alpha
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # get cls after the pooler, no dropout applied
        pooled_output = bert_out[1]
        pooled_output = self.cos_linear(pooled_output)
        rep_norms = pooled_output.norm(dim=1, keepdim=True)
        weight_norms = self.cos_linear.weight.data.norm(dim=1, keepdim=True)
        pooled_output.data /= rep_norms
        pooled_output.data /= weight_norms.t()
        pooled_output.data *= self.scale_alpha
        return pooled_output, None


class BertForSequenceTagging(LinearPredictor):
    """
    used for both tagging and ner.
    """

    def __init__(self, bert_config, out_dim, dropout=0.1):
        super(BertForSequenceTagging, self).__init__(bert_config, out_dim, dropout)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        if_tgts=None,
        **kwargs,
    ):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        bert_out = bert_out[0]
        bert_out = self.dropout(bert_out)
        bert_out = self.classifier(bert_out)
        logits = bert_out[if_tgts]
        return (
            logits,
            torch.argmax(bert_out, dim=-1, keepdim=False),
            bert_out,
        )
