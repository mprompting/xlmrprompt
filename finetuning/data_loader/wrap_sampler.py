from torch.utils.data import SequentialSampler, DataLoader, RandomSampler


def wrap_sampler(trn_batch_size, infer_batch_size, language, language_dataset):
    for split_name in ("trn_egs", "val_egs", "tst_egs"):
        egs = getattr(language_dataset, split_name)
        if len(egs) == 0:
            print(f"[WARN] {split_name} of {language} has zero egs")
        if split_name == "trn_egs":
            sampler = RandomSampler
            batch_size = trn_batch_size
        else:
            sampler = SequentialSampler
            batch_size = infer_batch_size
        dl = (
            DataLoader(egs, sampler=sampler(egs), batch_size=batch_size)
            if len(egs) > 0
            else None
        )
        setattr(language_dataset, split_name, dl)
    return language_dataset
