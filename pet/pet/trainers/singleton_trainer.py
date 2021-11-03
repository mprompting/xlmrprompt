import json
import os
import numpy as np
import torch
from typing import List, Dict
from pet.utils import InputExample, save_predictions, set_seed, find_free_port
from pet.modules.configs import WrapperConfig
from pet.modules.wrapper import TransformerModelWrapper
from pet.modules.configs import PREPROCESSORS
from pet.pvps.cs_patternization import PatternizedIterator
from pet.data_loaders.base import DataProcessor
from .configs import TrainConfig, EvalConfig, DDPConfig
from .singleton_evalfns import singleton_evaluate
import torch.distributed as dist
from multiprocessing import set_start_method
import copy
import log

try:
    set_start_method("spawn")
except RuntimeError:
    pass


logger = log.get_logger("root")


def init_model(
    config: WrapperConfig, train_config: WrapperConfig
) -> TransformerModelWrapper:
    # config: model_config
    assert (
        config.pattern_id is not None
    ), "A pattern_id must be set for initializing a new PET model"
    model = TransformerModelWrapper(config, train_config)
    return model


def train_single_model(
    model: TransformerModelWrapper,
    data_iterator: PatternizedIterator,
    config: TrainConfig,
    eval_config: EvalConfig = None,
):
    results_dict = {}
    model.model.to(config.device)
    global_step, tr_loss, best_score, best_model = model.train(
        -1,
        data_iterator,
        eval_config,
        config.device,
        n_gpu=config.n_gpu,
        num_train_epochs=config.num_train_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        adam_epsilon=config.adam_epsilon,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        lm_training=config.lm_training,
        use_logits=config.use_logits,
        alpha=config.alpha,
        temperature=config.temperature,
        max_steps=config.max_steps,
        ddp_config=None,
    )
    results_dict["global_step"] = global_step
    results_dict["average_loss"] = tr_loss
    return results_dict, best_score, best_model


def ddp_train_single_model(
    gpu,
    queue,
    finish_train,
    master_addr,
    master_port,
    model_config,
    train_config,
    eval_config,
    ddp_config,
    data_iterator,
):
    set_seed(train_config.seed)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    dist.init_process_group(backend="nccl", rank=gpu, world_size=ddp_config.world_size)

    list(map(lambda x: setattr(x, "device", gpu), [train_config, eval_config]))
    wrapper = init_model(model_config, train_config)
    preprocessor = PREPROCESSORS[model_config.wrapper_type](
        wrapper,
        model_config.task_name,
        pattern_lang=train_config.pattern_lang,
        pattern_id=model_config.pattern_id,
    )
    wrapper.preprocessor = preprocessor
    wrapper.model = wrapper.model.to(gpu)

    wrapper.model = torch.nn.parallel.DistributedDataParallel(
        wrapper.model,
        device_ids=[gpu],
        output_device=[gpu],
        find_unused_parameters=True,
    )

    data_iterator.ddp_patternize_trn(ddp_config, gpu)

    results_dict = {}
    global_step, tr_loss, best_score, best_model = wrapper.train(
        gpu,
        data_iterator,
        eval_config,
        gpu,
        n_gpu=train_config.n_gpu,
        num_train_epochs=train_config.num_train_epochs,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.learning_rate,
        adam_epsilon=train_config.adam_epsilon,
        warmup_steps=train_config.warmup_steps,
        max_grad_norm=train_config.max_grad_norm,
        lm_training=train_config.lm_training,
        use_logits=train_config.use_logits,
        alpha=train_config.alpha,
        temperature=train_config.temperature,
        max_steps=train_config.max_steps,
        ddp_config=ddp_config,
    )
    if gpu == 0:
        results_dict["global_step"] = global_step
        results_dict["average_loss"] = tr_loss
        queue.put((results_dict, best_score, best_model))
        print("Results put to queue")
    wrapper.model.cpu()
    finish_train.wait()


def _clean_name(module):
    state_dict = {}
    for name, val in module.items():
        state_dict[name.replace("module.", "")] = val
    return state_dict


def init_iter_module(model_config, train_config, eval_config, ddp_config, dataset):
    wrapper = init_model(model_config, train_config)
    if not ddp_config.do_ddp:
        train_batch_size = train_config.per_gpu_train_batch_size * train_config.n_gpu
        eval_batch_size = train_config.per_gpu_eval_batch_size * eval_config.n_gpu
    else:
        train_batch_size = train_config.per_gpu_train_batch_size
        eval_batch_size = train_config.per_gpu_train_batch_size * 2

    data_iterator = PatternizedIterator(
        dataset,
        wrapper.preprocessor,
        pattern_lang=train_config.pattern_lang,
        data_lang="en",
        trn_batch_size=train_batch_size,
        inf_batch_size=eval_batch_size,
    )
    data_iterator.patternize_trn()
    data_iterator.patternize_val()
    data_iterator.patternize_zs()
    return (data_iterator, wrapper)


def train_model_per_pattern(
    model_config: WrapperConfig,
    train_config: TrainConfig,
    eval_config: EvalConfig,
    ddp_config: DDPConfig,
    pattern_ids: List[int],
    output_dir: str,
    dataset: DataProcessor,
    repetitions: int = 1,
    do_train: bool = True,
    do_eval: bool = True,
):

    for pattern_id in pattern_ids:
        for iteration in range(repetitions):
            model_config.pattern_id = pattern_id
            results_dict = {}
            pattern_iter_output_dir = f"{output_dir}/p{pattern_id}-i{iteration}"
            if os.path.exists(pattern_iter_output_dir):
                logger.warning(
                    f"Path {pattern_iter_output_dir} already exists, skipping it..."
                )
                continue

            if not os.path.exists(pattern_iter_output_dir):
                os.makedirs(pattern_iter_output_dir)

            wrapper = None
            # Training
            if do_train:
                if ddp_config.do_ddp:
                    import torch.multiprocessing as mp

                    data_iterator, _ = init_iter_module(
                        model_config, train_config, eval_config, ddp_config, dataset
                    )
                    finish_train = mp.Event()
                    queue = mp.Queue()
                    ddp_config.world_size = ddp_config.num_nodes * ddp_config.num_ranks
                    master_port = find_free_port()
                    master_addr = "localhost"
                    processes = []
                    for gpu in range(ddp_config.num_ranks):
                        p = mp.Process(
                            target=ddp_train_single_model,
                            args=(
                                gpu,
                                queue,
                                finish_train,
                                master_addr,
                                master_port,
                                model_config,
                                train_config,
                                eval_config,
                                ddp_config,
                                data_iterator,
                            ),
                        )
                        p.start()
                        processes.append(p)
                    ddp_results = queue.get()
                    overalls = copy.deepcopy(ddp_results)
                    del ddp_results
                    finish_train.set()
                    for p in processes:
                        p.join()
                    print("Finish training", overalls[0])
                    trained_results, best_score, best_model = overalls
                else:
                    data_iterator, wrapper = init_iter_module(
                        model_config, train_config, eval_config, ddp_config, dataset
                    )
                    wrapper.rank_idx = torch.cuda.current_device()
                    trained_results, best_score, best_model = train_single_model(
                        wrapper,
                        data_iterator,
                        train_config,
                        eval_config,
                    )
                results_dict["best_score_dev"] = best_score
                with open(
                    os.path.join(pattern_iter_output_dir, "results.txt"), "w"
                ) as fh:
                    fh.write(str(results_dict))

                train_config.save(
                    os.path.join(pattern_iter_output_dir, "train_config.json")
                )
                eval_config.best_results = best_score
                eval_config.save(
                    os.path.join(pattern_iter_output_dir, "eval_config.json")
                )
                logger.info("Saving configs complete")
            if do_eval:
                logger.info("Starting evaluation...")
                if not wrapper:
                    wrapper = init_model(model_config, train_config)
                    best_model = _clean_name(best_model)
                    wrapper.model.load_state_dict(best_model)
                    if train_config.n_gpu > 1:
                        wrapper.model = torch.nn.DataParallel(wrapper.model)
                    wrapper.model.cuda()
                    wrapper.rank_idx = torch.cuda.current_device()
                _train_config = copy.deepcopy(train_config)
                _train_config.per_gpu_eval_batch_size = 4
                ddp_config.do_ddp = False
                test_data_iterator, _ = init_iter_module(
                    model_config, _train_config, eval_config, ddp_config, dataset
                )
                lang2evalscores = {}
                lang2predictions = {}
                for _eval_lang, _eval_loader in test_data_iterator.zs_iters.items():
                    logger.info(f"Eval on {_eval_lang}")
                    _eval_result = wrapper.eval(
                        _eval_loader, wrapper.rank_idx, zs_infer=True
                    )
                    _eval_result, zipped_preds = singleton_evaluate(
                        _eval_result, eval_config
                    )
                    save_predictions(
                        os.path.join(
                            pattern_iter_output_dir,
                            f"predictions,eval_data,{_eval_lang}.jsonl",
                        ),
                        wrapper,
                        _eval_result,
                    )
                    lang2evalscores[_eval_lang] = _eval_result["scores"]
                    lang2predictions[_eval_lang] = zipped_preds
                dev_2check_result = wrapper.eval(
                    test_data_iterator.val_iter, wrapper.rank_idx, zs_infer=True
                )
                dev_2check_result, _ = singleton_evaluate(
                    dev_2check_result, eval_config
                )
                save_predictions(
                    os.path.join(
                        pattern_iter_output_dir,
                        f"predictions,dev_2check_,en.jsonl",
                    ),
                    wrapper,
                    dev_2check_result,
                )
                logger.info(
                    f"--- RESULT (pattern_id={pattern_id}, iteration={iteration}) ---"
                )
                logger.info(f"Best dev score: {best_score}")
                logger.info(f"2check dev score: {dev_2check_result['scores']}")
                for _lang, _scores in lang2evalscores.items():
                    logger.info(f"{_lang}: {_scores}")
                results_dict["test_set_after_training_zstransfer"] = lang2evalscores
                with open(
                    os.path.join(pattern_iter_output_dir, "results.json"), "w"
                ) as fh:
                    json.dump(results_dict, fh)
                with open(
                    os.path.join(pattern_iter_output_dir, "zipped_lang2preds.json"), "w"
                ) as fh:
                    json.dump(lang2predictions, fh)
