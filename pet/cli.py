import argparse
import torch
import os
from typing import Tuple
from pet.data_loaders import (
    name2datasets,
    UNLABELED_SET,
    TRAIN_SET,
    DEV_SET,
    TEST_SET,
    METRICS,
    DEFAULT_METRICS,
)
from pet.modules.configs import WrapperConfig
from pet.utils import set_seed
import pet.trainers.singleton_trainer as singleton_trainer
import pet.trainers.configs as petconfig
import log

logger = log.get_logger("root")


def load_pet_configs(
    args,
) -> Tuple[
    WrapperConfig, petconfig.TrainConfig, petconfig.EvalConfig, petconfig.DDPConfig
]:
    # config for transformer models
    model_cfg = WrapperConfig(
        model_type=args.model_type,
        model_name_or_path=args.model_name_or_path,
        wrapper_type=args.wrapper_type,
        task_name=args.task_name,
        label_list=args.label_list,
        max_seq_length=args.pet_max_seq_length,
        cache_dir=args.cache_dir,
    )
    # pet training config
    train_cfg = petconfig.TrainConfig(
        pattern_lang=args.pattern_lang,
        device=args.device,
        per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
        per_gpu_unlabeled_batch_size=-1,
        n_gpu=args.n_gpu,
        num_train_epochs=args.pet_num_train_epochs,
        max_steps=args.pet_max_steps,
        gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lm_training=args.lm_training,
        alpha=args.alpha,
        temperature=args.temperature,
        seed=args.seed,
    )
    # evaluation config
    eval_cfg = petconfig.EvalConfig(
        device=args.device,
        n_gpu=args.n_gpu,
        metrics=args.metrics,
        per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
        decoding_strategy=args.decoding_strategy,
    )
    ddp_cfg = petconfig.DDPConfig(
        do_ddp=args.do_ddp, num_ranks=args.num_ranks, num_nodes=args.num_nodes
    )
    return (model_cfg, train_cfg, eval_cfg, ddp_cfg)


def main(args):

    logger.info("Experiment Parameters: {}".format(args))

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir
            )
        )

    set_seed(args.seed)
    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in name2datasets:
        raise ValueError("Task '{}' not found".format(args.task_name))

    # trn_egs, val_egs, zs_egs, List[InputExample]
    data_lang = args.data_lang if args.data_lang else None
    dataset = name2datasets[args.task_name](args.num_shots, data_lang)

    # standard label list before verbalizing
    args.label_list = dataset.get_labels()

    # init eval metrics and configs
    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)
    # split the args for transformers, training, evaluating
    pet_model_cfg, pet_train_cfg, pet_eval_cfg, ddp_cfg = load_pet_configs(args)

    singleton_trainer.train_model_per_pattern(
        pet_model_cfg,
        pet_train_cfg,
        pet_eval_cfg,
        ddp_cfg,
        dataset=dataset,
        pattern_ids=args.pattern_ids,
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
    )


if __name__ == "__main__":
    from parameters import get_args

    args = get_args()
    main(args)
