### cluster the environments using the Dirichlet mixture with CRP
python env_clustering.py --env HalfCheetahVel-v1 --model_path saves/cheetah \
    --et_length 1 --device cuda:0 --num_periods 50 --seed 520

### train the polcies according the built library
python policy_training.py --env HalfCheetahVel-v1 --model_path saves/cheetah \
    --output output/cheetah --num_iter 500 --algorithm ppo --opt adam \
    --lr 1e-3 --baseline linear --device cuda:0 --num_periods 50 --seed 520

