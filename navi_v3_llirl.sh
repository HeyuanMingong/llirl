### cluster the environments using the Dirichlet mixture with CRP
python env_clustering.py --env Navigation2D-v3 --model_path saves/navi_v3 \
    --et_length 1 --device cpu --num_periods 50 --seed 20210128

### train the polices according the built library
python policy_training.py --env Navigation2D-v3 --model_path saves/navi_v3 \
    --output output/navi_v3 --algorithm reinforce --opt sgd --lr 0.02 \
    --num_iter 100 --num_periods 50 --device cpu --seed 20210128
