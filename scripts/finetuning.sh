#!/bin/bash

declare -a list_of_num_shots=(1 2 4 8 16 32 64 128 256)
declare -a list_of_seeds=(1 2 3 4 5)

for ((which_seed=0;which_seed<${#list_of_seeds[@]};++which_seed)); do
    for ((which_num_shots=0;which_num_shots<${#list_of_num_shots[@]};++which_num_shots)); do
        python finetuning_baseline.py --override False \
            --experiment test \
            --ptl xlm-roberta \
            --model xlm-roberta-base \
            --dataset_name xnli \
            --trn_languages english \
            --eval_language arabic,bulgarian,german,greek,english,spanish,french,hindi,russian,swahili,thai,turkish,urdu,vietnamese,chinese \
            --finetune_epochs 50 \
            --finetune_lr 1e-5 \
            --finetune_batch_size 8 \
            --inference_batch_size 256 \
            --num_shots ${list_of_num_shots[which_num_shots]} \
            --manual_seed ${list_of_seeds[which_seed]} \
            --max_seq_len 256 \
            --train_fast True \
            --world 1,2,3,4
    done
done