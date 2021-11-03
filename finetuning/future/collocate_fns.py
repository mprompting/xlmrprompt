import torch


def seqcls_collocate(batched, device=None):
    if device is None:
        device = torch.cuda.current_device()
    uids, input_ids, golds, attention_mask, token_type_ids = batched
    batched = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "token_type_ids": token_type_ids.to(device),
    }
    return (batched, golds.to(device), uids, None)


task2collocate_fn = {
    "marc": seqcls_collocate,
    "mldoc": seqcls_collocate,
    "conll2003": None,
    "argustan": seqcls_collocate,
    "pawsx": seqcls_collocate,
    "xnli": seqcls_collocate,
    "panx": None,
    "udpos": None,
}
