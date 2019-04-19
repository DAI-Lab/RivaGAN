CUDA_VISIBLE_DEVICES=0 screen -dm python3 train.py --experiment product --use_jpeg 0
sleep 30s
CUDA_VISIBLE_DEVICES=1 screen -dm python3 train.py --experiment product --use_jpeg 1
sleep 30s
CUDA_VISIBLE_DEVICES=2 screen -dm python3 train.py --experiment product-positional --use_jpeg 0
sleep 30s
CUDA_VISIBLE_DEVICES=3 screen -dm python3 train.py --experiment product-positional --use_jpeg 1
