source activate pytorch
export CUDA_VISIBLE_DEVICES=0

python train.py --use_critic 0 --use_adversary 0 --use_crop 0 --use_scale 0 --use_compression 0
python train.py --use_critic 1 --use_adversary 0 --use_crop 0 --use_scale 0 --use_compression 0
python train.py --use_critic 0 --use_adversary 1 --use_crop 0 --use_scale 0 --use_compression 0
python train.py --use_critic 1 --use_adversary 1 --use_crop 0 --use_scale 0 --use_compression 0
