#!/bin/bash
# bash scripts/baseline_regression.bash

# data_types=('isoDEH' 'cycDEH' 'TPD')
# models=('KNeighborsRegressor' 'DecisionTreeRegressor' 'SVR' 'RandomForestRegressor' 'GradientBoostingRegressor' 'BayesianRidge' 'LinearRegressor' 'LassoRegressor' )
# seeds=(1 2 3)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --seed $seed --model $model \
#         --epochs 200 --device 2 --data_mode 'one-to-one' 
# done
# done
# done

# data_types=('isoDEH' 'cycDEH')
# models=('KNeighborsRegressor' 'DecisionTreeRegressor' 'SVR' 'RandomForestRegressor' 'GradientBoostingRegressor' 'BayesianRidge' 'LinearRegressor' 'LassoRegressor' )

# seeds=(3)
# for data_type in "${data_types[@]}"; do
# for seed in "${seeds[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --seed $seed --model $model \
#         --epochs 200 --device 6 --data_mode 'one-to-one' 
# done
# done
# done




# data_types=('cycDEH')
# models=('KNeighborsRegressor' 'DecisionTreeRegressor' 'SVR' 'RandomForestRegressor' 'GradientBoostingRegressor' 'BayesianRidge' 'LinearRegressor' 'LassoRegressor' 'RidgeRegressor')
# for data_type in "${data_types[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 0 --data_mode 'one-to-one'
# done
# done


# data_types=('isoDEH' 'TPD')
# models=('RidgeRegressor')
# for data_type in "${data_types[@]}"; do
# for model in "${models[@]}"; do
#         python main.py --dataset $data_type --model $model \
#         --epochs 200 --device 0 --data_mode 'one-to-one'
# done
# done


# get shap
# data_types=('isoDEH' 'cycDEH' 'TPD')
data_types=('cycDEH')
models=('KNeighborsRegressor' 'DecisionTreeRegressor' 'SVR' 'RandomForestRegressor' 'GradientBoostingRegressor' 'BayesianRidge' 'LinearRegressor' 'LassoRegressor' 'RidgeRegressor')

seeds=(3)
for data_type in "${data_types[@]}"; do
for seed in "${seeds[@]}"; do
for model in "${models[@]}"; do
        python main.py  --dataset $data_type --seed $seed --model $model \
        --epochs 200 --device 2 --data_mode 'one-to-one' 
done
done
done


