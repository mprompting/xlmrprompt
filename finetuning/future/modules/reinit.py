# -*- coding: utf-8 -*-
from torch import nn
import transformers


def get_ptl(conf):
    if conf.ptl == "bert":
        return getattr(transformers, "modeling_bert")
    elif conf.ptl == "distilbert":
        return getattr(transformers, "modeling_distilbert")
    elif conf.ptl == "modeling_roberta":
        return getattr(transformers, "modeling_roberta")
    else:
        raise NotImplementedError("invalid ptl.")


# initialize the weights, following the transformer package
def _init_weights(conf, name, module):
    print_info = lambda info: print(f"{info} module name: {name}")
    if isinstance(module, nn.Embedding):
        print_info("reinit embedding layer.")
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, get_ptl(conf).BertLayerNorm):
        print_info("reinit layer-norm.")
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear):
        flag = False
        if "attention.self" in name:
            print_info("reinit linear for attention.self.")
            flag = True
        if "attention.output" in name:
            print_info("reinit linear for attention.output.")
            flag = True
        if "intermediate" in name:
            print_info("reinit linear for intermediate.")
            flag = True
        if "output" in name and "attention" not in name:
            print_info("reinit linear for layer output.")
            flag = True
        if "pooler" in name:
            print_info("reinit linear for pooler.")
            flag = True
        # I wont re-init the classifier layer since its untrained
        if flag:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


def random_reinit_ptl(conf, model):
    for _name, _module in model.named_modules():
        _init_weights(conf, _name, _module)
    return model
