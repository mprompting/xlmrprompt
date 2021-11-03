# -*- coding: utf-8 -*-
from os.path import join
import argparse


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint_finetuning")

    parser = argparse.ArgumentParser()
    parser.add_argument("--override", type=str2bool, default=True)

    parser.add_argument("--experiment", type=str, default="debug")
    parser.add_argument("--ptl", type=str, default="bert")
    parser.add_argument("--model", type=str, default="bert-base-uncased")

    parser.add_argument("--dataset_name", type=str, default="mldoc")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--trn_languages", type=str, default="japanese")
    parser.add_argument("--eval_languages", type=str, default="english")

    # supervised finetuning setup
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--eval_every_batch", type=int, default=10)
    parser.add_argument("--finetune_batch_size", type=int, default=32)
    parser.add_argument("--finetune_lr", type=float, default=5e-5)

    # speeding up inference
    parser.add_argument("--inference_batch_size", type=int, default=512)

    # miscs
    parser.add_argument("--data_path", default=RAW_DATA_DIRECTORY, type=str)
    parser.add_argument("--checkpoint", default=TRAINING_DIRECTORY, type=str)
    parser.add_argument("--manual_seed", type=int, default=42, help="manual seed")
    parser.add_argument("--train_fast", default=True, type=str2bool)
    parser.add_argument("--world", default="0", type=str)

    parser.add_argument("--num_shots", type=int, default=-1)
    parser.add_argument("--in_lang", type=str)

    return parser


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = get_args()
    conf = parser.parse_args()
