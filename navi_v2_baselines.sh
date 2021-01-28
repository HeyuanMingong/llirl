### run the baseline approaches including: CA, Robust, Adaptive, MAML
python baselines.py --env Navigation2D-v2 --model_path saves/navi_v2 --output output/navi_v2 \
    --algorithm reinforce --opt sgd --lr 0.02 --num_iter 100 --num_periods 50 \
    --device cuda:0 --seed 931009 --ca --robust --adaptive --maml

