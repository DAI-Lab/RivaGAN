source activate pytorch
export CUDA_VISIBLE_DEVICES=2

python train.py --data_dim 32 --use_noise 1 --use_critic 1 --use_adversary 0
python train.py --data_dim 64 --use_noise 1 --use_critic 1 --use_adversary 0
