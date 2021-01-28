### cluster the environments using the Dirichlet mixture with CRP
python env_clustering.py --env AntVel-v1 --model_path saves/ant \
    --et_length 3 --device cuda:0 --num_periods 50 --seed 111

### train the policies according to the built library
python policy_training.py --env AntVel-v1 --model_path saves/ant \
    --output output/ant --algorithm ppo --opt adam --lr 1e-3 --num_iter 500 \
    --baseline linear --num_periods 50 --device cuda:0 --seed 111
