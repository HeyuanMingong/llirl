python baselines.py --env AntVel-v1 --model_path saves/ant --output output/ant \
    --num_iter 500 --lr 1e-3 --opt adam --algorithm ppo --num_periods 50 \
    --baseline linear --device cuda:0 --seed 111 --ca --robust --adaptive --maml

