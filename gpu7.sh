source activate pytorch_p36
export CUDA_VISIBLE_DEVICES=7

python train.py --use_critic 1 --use_adversary 0 --use_jpeg 1 --use_crop 1 --use_scale 1
python train.py --use_critic 0 --use_adversary 1 --use_jpeg 1 --use_crop 1 --use_scale 1
