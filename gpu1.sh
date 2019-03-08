source activate pytorch
export CUDA_VISIBLE_DEVICES=1

python train.py --use_compression 1
python train.py --use_compression 1 --use_crop 1
python train.py
