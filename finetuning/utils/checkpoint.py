# -*- coding: utf-8 -*-
import os
import time
import torch

from data_loader.data_configs import language2abbre


def init_checkpoint_baseline(conf):
    invalid = True
    trn_languages = "-".join([language2abbre[l] for l in conf.trn_languages])
    # if too many languages, we just report the number of languages, otherwise file length > 255
    val_languages = (
        "-".join([language2abbre[l] for l in conf.eval_languages])
        if len(conf.eval_languages) <= 15
        else len(conf.eval_languages)
    )
    while invalid:
        time_id = str(int(time.time()))
        conf.time_stamp_ = (
            f"{time_id}_model_task-{conf.task}_"
            f"flr-{conf.finetune_lr:.1E}_"
            f"ftbs-{conf.finetune_batch_size}_"
            f"ftepcs-{conf.finetune_epochs}_"
            f"sd-{conf.manual_seed}_"
            f"trnfast-{conf.train_fast}_"
            f"evalevery-{conf.eval_every_batch}_"
            f"tlang-{trn_languages}_"
            f"vlang-{val_languages}_"
            f"nshots-{conf.num_shots}"
        )
        conf.checkpoint_root = os.path.join(
            conf.checkpoint,
            conf.task,
            conf.experiment if conf.experiment is not None else "",
            conf.time_stamp_,
        )

        if not os.path.exists(conf.checkpoint_root) and build_dirs(
            conf.checkpoint_root
        ):
            invalid = False

    print(conf.checkpoint_root)
    assert len(os.path.abspath(conf.checkpoint_root)) < 255
    return conf.checkpoint_root


def init_checkpoint_adapt(conf):
    invalid = True
    adapt_trn_languages = "-".join(
        [language2abbre[l] for l in conf.adapt_trn_languages]
    )
    while invalid:
        time_id = str(int(time.time()))
        conf.time_stamp_ = (
            f"{time_id}_model_task-{conf.task}_"
            f"adtlr{conf.adapt_lr:.1E}_"
            f"adepcs-{conf.adapt_epochs}_"
            f"adtbs-{conf.adapt_batch_size}_"
            f"sd-{conf.manual_seed}_"
            f"trnfast-{conf.train_fast}_"
            f"adshots-{conf.adapt_num_shots}_"
            f"adtrnl-{adapt_trn_languages}"
        )
        conf.checkpoint_root = os.path.join(
            conf.checkpoint,
            conf.task,
            conf.experiment if conf.experiment is not None else "",
            conf.time_stamp_,
        )

        if not os.path.exists(conf.checkpoint_root) and build_dirs(
            conf.checkpoint_root
        ):
            invalid = False

    print(conf.checkpoint_root)
    assert len(os.path.abspath(conf.checkpoint_root)) < 255
    return conf.checkpoint_root


def init_checkpoint_inference(conf):
    invalid = True
    adapt_trn_languages = "-".join(
        [language2abbre[l] for l in conf.adapt_trn_languages]
    )
    while invalid:
        time_id = str(int(time.time()))
        conf.time_stamp_ = (
            f"{time_id}_model_task-{conf.task}_"
            f"sd-{conf.manual_seed}_"
            f"trnfast-{conf.train_fast}_"
            f"adshots-{conf.adapt_num_shots}_"
            f"adtrnl-{adapt_trn_languages}"
        )
        conf.checkpoint_root = os.path.join(
            conf.checkpoint,
            conf.task,
            conf.experiment if conf.experiment is not None else "",
            conf.time_stamp_,
        )

        if not os.path.exists(conf.checkpoint_root) and build_dirs(
            conf.checkpoint_root
        ):
            invalid = False

    print(conf.checkpoint_root)
    assert len(os.path.abspath(conf.checkpoint_root)) < 255
    return conf.checkpoint_root


def _save_to_checkpoint(state, dirname, filename):
    checkpoint_path = os.path.join(dirname, filename)
    torch.save(state, checkpoint_path)
    return checkpoint_path


def build_dirs(path):
    try:
        os.makedirs(path)
        return True
    except Exception as e:
        print(" encounter error: {}".format(e))
        return False
