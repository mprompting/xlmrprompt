import torch


def seqcls_batch_to_device(batched, cuda_index=0):
    golds, input_ids, attention_mask, token_type_ids = map(
        lambda x: x.cuda(cuda_index), batched
    )
    return (
        golds,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
        None,
    )


class _Seqcls_task_container(torch.utils.data.Dataset):
    """NOTE: __getitem__ must be consistent with ``seqcls_batch_to_device``"""

    def __init__(self, meta_entries):
        self.input_idses = torch.as_tensor(
            [entry.content_tensor.input_ids for entry in meta_entries], dtype=torch.long
        )
        self.golds = torch.as_tensor(
            [entry.gold_label for entry in meta_entries], dtype=torch.long
        )
        self.attention_maskes = torch.as_tensor(
            [entry.content_tensor.attention_mask for entry in meta_entries],
            dtype=torch.long,
        )
        self.token_type_idses = torch.as_tensor(
            [entry.content_tensor.token_type_ids for entry in meta_entries],
            dtype=torch.long,
        )

    def __len__(self):
        return self.golds.shape[0]

    def __getitem__(self, idx):
        return (
            self.golds[idx],
            self.input_idses[idx],
            self.attention_maskes[idx],
            self.token_type_idses[idx],
        )

