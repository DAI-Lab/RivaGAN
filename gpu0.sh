source activate pytorch
export CUDA_VISIBLE_DEVICES=0

python train.py --data_dim 32 --use_noise 0 --use_critic 0 --use_adversary 0
python train.py --data_dim 64 --use_noise 0 --use_critic 0 --use_adversary 0

python train.py --data_dim 32 --use_classic 1
python train.py --data_dim 64 --use_classic 1
