python baselines.py --env HalfCheetahVel-v1 --model_path saves/cheetah --output output/cheetah \
    --algorithm ppo --opt adam --lr 1e-3 --num_iter 500 --num_periods 50 \
    --device cuda:0 --baseline linear --seed 520 --ca --robust --adaptive --maml



