from os.path import join
from pet.modules.configs import WRAPPER_TYPES, MODEL_CLASSES
from pet.data_loaders import name2datasets
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Command line PET")

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the data files for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        choices=MODEL_CLASSES.keys(),
        help="The type of the pretrained language model to use",
    )  # bert
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to the pre-trained model or shortcut name",
    )  # bert-base-uncased
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        choices=name2datasets.keys(),
        help="The name of the task to train/evaluate on",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written",
    )

    # PET-specific optional parameters
    parser.add_argument(
        "--wrapper_type",
        default="mlm",
        choices=WRAPPER_TYPES,
        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
        "for a permuted language model like XLNet (only for PET)",
    )
    parser.add_argument(
        "--pattern_ids",
        default=[0],
        type=int,
        nargs="+",
        help="The ids of the PVPs to be used (only for PET)",
    )
    parser.add_argument(
        "--lm_training",
        action="store_true",
        help="Whether to use language modeling as auxiliary task (only for PET)",
    )
    parser.add_argument(
        "--alpha",
        default=0.9999,
        type=float,
        help="Weighting term for the auxiliary language modeling task (only for PET)",
    )
    parser.add_argument(
        "--temperature",
        default=2,
        type=float,
        help="Temperature used for combining PVPs (only for PET)",
    )
    parser.add_argument(
        "--reduction",
        default="mean",
        choices=["wmean", "mean"],
        help="Reduction strategy for merging predictions from multiple PET models. Select either "
        "uniform weighting (mean) or weighting based on train set accuracy (wmean)",
    )
    parser.add_argument(
        "--decoding_strategy",
        default="default",
        choices=["default", "ltr", "parallel"],
        help="The decoding strategy for PET with multiple masks (only for PET)",
    )
    parser.add_argument(
        "--pet_repetitions",
        default=3,
        type=int,
        help="The number of times to repeat PET training and testing with different seeds.",
    )
    parser.add_argument(
        "--pet_max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--pet_per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for PET training.",
    )
    parser.add_argument(
        "--pet_per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for PET evaluation.",
    )
    parser.add_argument(
        "--pet_gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass in PET.",
    )
    parser.add_argument(
        "--pet_num_train_epochs",
        default=3,
        type=float,
        help="Total number of training epochs to perform in PET.",
    )
    parser.add_argument(
        "--pet_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where to store the pre-trained models downloaded from S3.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
        help="Weight decay if we apply some.",
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to perform training"
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to perform evaluation"
    )

    # languge specifications
    parser.add_argument(
        "--pattern_lang", type=str, required=True,
    )
    # DDP related
    parser.add_argument("--do_ddp", action="store_true")
    parser.add_argument(
        "--num_ranks", type=int, default=1, help="number of cards per node"
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes")

    # shots
    parser.add_argument(
        "--num_shots", type=int, default=-1,
    )
    parser.add_argument("--data_lang", type=str)

    args = parser.parse_args()
    return args
