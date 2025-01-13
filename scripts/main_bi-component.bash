#!/bin/bash
# bash scripts/main_bi-component.bash

# data_types=('isoDEH' 'cycDEH' 'TPD')
# models=('TR')
# for data_type in "${data_types[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 0 --data_mode 'multi-to-one'
# done
# done


# data_types=('bi-isoDEH')
# models=('TransformerRegressor') 
# # models=('TransformerRegressor' 'Attention')
# seeds=(1)
# for data_type in "${data_types[@]}"; do
# for model in "${models[@]}"; do
# for seed in "${seeds[@]}"; do

#         python main_bi-component.py --dataset $data_type --model $model \
#         --epochs 400 --device 0 --data_mode 'one-to-one' --seed $seed --lr 0.0005 --batch_size 8
# done
# done
# done


# data_types=('bi-cycDEH')
# # models=('Attention') 
# models=('TransformerRegressor')
# seeds=(1)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main_bi-component.py --dataset $data_type --model $model \
#         --epochs 400 --device 1 --data_mode 'one-to-one' --seed $seed --lr 0.0005 --batch_size 8
# done
# done
# done
 
data_types=('bi-TPD')
models=('TransformerRegressor') 
# models=('TransformerRegressor' 'Attention')
seeds=(1)
for data_type in "${data_types[@]}"; do
for seed in "${seeds[@]}"; do
for model in "${models[@]}"; do
        python main_bi-component.py --dataset $data_type --model $model \
        --epochs 400 --device 0 --data_mode 'one-to-one' --seed $seed --lr 0.0015 --batch_size 16
done
done
done

# data_types=('bi-TPD' 'bi-isoDEH')
# models=('Attention')
# seeds=(2 3)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main_bi-component.py --dataset $data_type --model $model \
#         --epochs 400 --device 0 --data_mode 'one-to-one' --seed $seed --batch_size 8  --lr 0.0015
# done
# done
# done
