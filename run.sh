CUDA_VISIBLE_DEVICES=0 screen -dm python3 train.py --experiment default
sleep 30s
CUDA_VISIBLE_DEVICES=1 screen -dm python3 train.py --experiment positional
sleep 30s
CUDA_VISIBLE_DEVICES=2 screen -dm python3 train.py --experiment small
sleep 30s
CUDA_VISIBLE_DEVICES=3 screen -dm python3 train.py --experiment large
