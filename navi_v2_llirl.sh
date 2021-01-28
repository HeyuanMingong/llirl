### cluster the environments using the Dirichlet mixture with CRP
python env_clustering.py --env Navigation2D-v2 --model_path saves/navi_v2 \
    --et_length 1 --device cpu --num_periods 50 --seed 931009

### train the policies according to the built library
python policy_training.py --env Navigation2D-v2 --model_path saves/navi_v2 \
    --output output/navi_v2 --algorithm reinforce --opt sgd --lr 0.02 \
    --num_iter 100 --num_periods 50 --device cpu --seed 931009

