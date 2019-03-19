source activate pytorch_p36
export CUDA_VISIBLE_DEVICES=4

python train.py --use_critic 1 --use_adversary 0 --use_jpeg 0 --use_crop 0 --use_scale 0
python train.py --use_critic 0 --use_adversary 1 --use_jpeg 0 --use_crop 0 --use_scale 0
