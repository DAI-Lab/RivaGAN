import os
import cv2
import torch
from glob import glob
from random import randint
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):

    def __init__(self, root_dir, seq_len=10, center_crop=(128, 128)):
        self.seq_len = seq_len
        self.center_crop = center_crop
        
        self.videos = []
        for path in glob(os.path.join(root_dir, "*.avi")):
            cap = cv2.VideoCapture(path)
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.videos.append((path, nb_frames))

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        path, nb_frames = self.videos[idx]
        start_idx = randint(0, nb_frames - self.seq_len - 1)
        
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx-1)
        ok, frame = cap.read()
        H, W, D = frame.shape
        dx, dy = self.center_crop # target size
        x = randint(0, W-dx-1)
        y = randint(0, H-dy-1)
        
        frames = []
        for _ in range(self.seq_len):
            ok, frame = cap.read()
            frame = frame[y:y+dy,x:x+dx]
            frames.append(frame / 127.5 - 1.0)
        x = torch.FloatTensor(frames)
        x = x.permute(3, 0, 1, 2)
        return x

def load_train_val(seq_len, batch_size):
    train = DataLoader(VideoDataset(
        "data/hollywood2/train", 
        center_crop=(128, 128), 
        seq_len=seq_len,
    ), shuffle=True, num_workers=8, batch_size=batch_size)
    val = DataLoader(VideoDataset(
        "data/hollywood2/val", 
        center_crop=(128, 128),
        seq_len=seq_len,
    ), shuffle=False, num_workers=8, batch_size=batch_size)
    return train, val
