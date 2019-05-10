source activate pytorch
export CUDA_VISIBLE_DEVICES=3

python train.py --data_dim 32 --use_noise 1 --use_critic 1 --use_adversary 1
python train.py --data_dim 64 --use_noise 1 --use_critic 1 --use_adversary 1
