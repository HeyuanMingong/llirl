python baselines.py --env HopperVel-v1 --model_path saves/hopper --output output/hopper \
    --algorithm ppo --opt adam --lr 1e-3 --num_iter 200 --num_periods 50 \
    --device cuda:0 --baseline linear --seed 418 --ca --robust --adaptive --maml



