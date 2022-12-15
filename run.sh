#!/usr/bin/env bash
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


model=resnet
lr=0.0002
log_type=txt
dataset=caltech101
i=0
active_method=ntk_memo
label=max


CUDA_VISIBLE_DEVICES=0 python src/main.py --seed $i --log_type $log_type \
--base_model $model --dataset_str $dataset --test_interval 5 --final_extra_epochs 30 \
--wo_train_sim 0 --norm_inf 0  --wo_de_inf 1 --inf_type full \
--active_method $active_method --init_label_perC 5 --budget_num_per_query 500 --grad_update_size 50 \
--total_query_times 15 --starting_epoch 1 --fixed_query_interval 0 \
--query_interval 140 100 80 80 70 70 70 70 70 70 70 70 70 70 70 \
--batch_size 200 --small_bz 50 --lr $lr  --pesudo $label