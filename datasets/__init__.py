import os
import cv2
import torch
from glob import glob
from random import randint
from torch.utils.data import Dataset, DataLoader

class VideoDataset(Dataset):

    def __init__(self, root_dir, mode="train", random_crop=False, seq_len=10):
        self.mode = mode
        self.samples = []
        self.seq_len = seq_len
        self.random_crop = random_crop
        for path in glob(os.path.join(root_dir, "*.avi")):
            cap = cv2.VideoCapture(path)
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(0, nb_frames - self.seq_len):
                self.samples.append((path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i = self.samples[idx]
        
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)

        frames = []
        for _ in range(self.seq_len):
            ok, frame = cap.read()
            frames.append(frame)
            assert ok

        x = torch.FloatTensor(frames)
        x = x.permute(0, 3, 1, 2)

        # random crop
        if self.random_crop:
            w, h = 300, 200
            i = randint(h//2, x.size(2)-h//2-1)
            j = randint(w//2, x.size(3)-w//2-1)
            x = x[:,:,i-h//2:i+h//2,j-w//2:j+w//2]

        return x / 127.5 - 1.0

def load_train_val():
    train = DataLoader(VideoDataset(
        "datasets/hollywood2/train", 
        random_crop=True, 
        seq_len=5
    ), shuffle=True, num_workers=32)
    val = DataLoader(VideoDataset(
        "datasets/hollywood2/val", 
        random_crop=True, 
        seq_len=5
    ), num_workers=32)
    return train, val
