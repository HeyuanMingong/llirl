### run the baseline approaches including: CA, Robust, Adaptive, MAML
python baselines.py --env Navigation2D-v1 --model_path saves/navi_v1 --output output/navi_v1 \
    --algorithm reinforce --opt sgd --lr 0.02 --num_iter 100 --num_periods 50 \
    --device cuda:0 --seed 1009 --ca --robust --adaptive --maml


