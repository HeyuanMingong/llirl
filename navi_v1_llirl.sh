### Cluster the environments using the Dirichlet mixture model with CRP
python env_clustering.py --env Navigation2D-v1 --model_path saves/navi_v1 \
    --et_length 1 --num_periods 50 --device cpu --seed 1009

### Train the policies accroding to built library
python policy_training.py --env Navigation2D-v1 --model_path saves/navi_v1 \
    --output output/navi_v1 --algorithm reinforce --opt sgd --lr 0.02 \
    --num_iter 100 --num_periods 50 --device cpu --seed 1009


