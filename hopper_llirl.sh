### cluster the environments using the Dirichlet mixture with CRP
python env_clustering.py --env HopperVel-v1 --model_path saves/hopper \
    --et_length 3 --device cuda:0 --num_periods 50 --seed 418

### train the policies according to built library
python policy_training.py --env HopperVel-v1 --model_path saves/hopper \
    --output output/hopper --algorithm ppo --opt adam --lr 1e-3 --baseline linear \
    --num_iter 200 --device cuda:0 --num_periods 50 --seed 418


