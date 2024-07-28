from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import LataccelTokenizer


class LatAccelDataset(Dataset):
    def __init__(self, data):
        assert data.ndim == 3
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :, :-1], self.data[idx, :, -1].to(dtype=torch.int64)


class LatAccelDataModule(pl.LightningDataModule):
    x_cols = [
        "steerFiltered",
        "roll",
        "vEgo",
        "aEgo",
    ]
    y_col = "latAccelLocalizer"
    SEGMENT_LENGTH = 600

    def __init__(self, path: str, batch_size: int):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = LataccelTokenizer()

    def setup(self, stage: str = None):
        segments = []
        for file in glob(f"{self.path}*.csv"):
            df = pd.read_csv(file)
            df = df[self.x_cols + [self.y_col]]
            df["roll"] = np.sin(df["roll"]) * 9.81
            df[self.y_col] = self.tokenizer.encode(df[self.y_col])
            val = df.values
            if val.shape[0] < self.SEGMENT_LENGTH:
                # pad with nan
                val = np.pad(val, ((0, self.SEGMENT_LENGTH - val.shape[0]), (0, 0)))
            elif val.shape[0] > self.SEGMENT_LENGTH:
                # truncate
                val = val[:self.SEGMENT_LENGTH]
            # add batch dimension
            val = val[np.newaxis]
            segments.append(val)

        # Concatenate: (n_sequences, sequence_length, n_features)
        data = np.concatenate(segments, axis=0)

        train_size = int(data.shape[0] * 0.8)
        self.train = LatAccelDataset(data[:train_size])
        self.val = LatAccelDataset(data[train_size:])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
