source activate pytorch
export CUDA_VISIBLE_DEVICES=1

python train.py --use_compression 0 --use_critic 0 --use_adversary 1
python train.py --use_compression 1 --use_critic 0 --use_adversary 1
python train.py --use_compression 0 --use_critic 1 --use_adversary 0
python train.py --use_compression 1 --use_critic 1 --use_adversary 0
