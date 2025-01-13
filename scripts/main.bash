#!/bin/bash
# bash scripts/main.bash

# data_types=('isoDEH' 'cycDEH' 'TPD')
# models=('TR')
# for data_type in "${data_types[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 0 --data_mode 'multi-to-one'
# done
# done

# data_types=('isoDEH' 'cycDEH' 'TPD')
# models=('TR' 'Attention')
# seeds=(1 2 3 )
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 3 --data_mode 'one-to-one' --seed $seed --batch_size 5
# done
# done
# done

# data_types=('isoDEH' 'cycDEH' 'TPD')
# data_types=('isoDEH' 'cycDEH')
# # data_types=('TPD')
# models=('Attention')
# seeds=(1 2 3 )
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 4 --data_mode 'one-to-one' --seed $seed --batch_size 8 --lr 0.002
# done
# done
# done

# data_types=('isoDEH'   'TPD')
# # data_types=('cycDEH')
# models=('TR' )
# seeds=(3 )
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 2 --data_mode 'multi-to-one' --seed $seed --batch_size 8 --lr 0.002
# done
# done
# done


# data_types=('isoDEH' 'cycDEH' 'TPD')


# data_types=('TPD')
# models=('TransformerRegressor' ) #TransformerRegressor
# seeds=(1 2 3)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 400 --device 2 --data_mode 'one-to-one' --seed $seed --batch_size 16 --lr 0.0005
# done
# done
# done

# data_types=('isoDEH')
# models=('TransformerRegressor' ) #TransformerRegressor
# seeds=(1 2 3)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 400 --device 2 --data_mode 'one-to-one' --seed $seed --batch_size 32 --lr 0.0008
# done
# done
# done

data_types=('cycDEH' )
models=('TransformerRegressor' ) #TransformerRegressor
seeds=(1 2 3)
for data_type in "${data_types[@]}"; do
for seed in "${seeds[@]}"; do
for model in "${models[@]}"; do
        python main.py --dataset $data_type --model $model \
        --epochs 400 --device 2 --data_mode 'one-to-one' --seed $seed --batch_size 32 --lr 0.0008
done
done
done