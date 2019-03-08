source activate pytorch
export CUDA_VISIBLE_DEVICES=0

python train.py --use_jpeg_compression 1
python train.py --use_jpeg_compression 1 --use_crop 1
python train.py --use_jpeg_compression 1 --use_critic 1
python train.py --use_jpeg_compression 1 --use_crop 1 --use_critic 1
