import logging
from glob import glob

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class LatAccelDataset(Dataset):
    def __init__(self, dfs: list[pd.DataFrame], input_length, x_cols: list[str], y_col: str, segment_length: int):
        self.data = dfs
        self.input_length = input_length
        self.x_cols = x_cols
        self.y_col = y_col
        self.segment_length = segment_length

    def __len__(self):
        return len(self.data) * (self.segment_length - self.input_length)

    def __getitem__(self, idx):
        segment_idx = idx // (self.segment_length - self.input_length)
        start_idx = idx % (self.segment_length - self.input_length)
        segment = self.data[segment_idx].iloc[start_idx:start_idx + self.input_length]
        x = torch.tensor(segment[self.x_cols].values, dtype=torch.float32)
        y = torch.tensor(segment[self.y_col].values, dtype=torch.float32)
        return x, y


class LatAccelDataModule(pl.LightningDataModule):
    def __init__(self, path: str, x_cols: list[str], y_col: str, batch_size: int, input_length: int, segment_length: int = 599):
        super().__init__()
        self.path = path
        self.x_cols = x_cols
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_length = input_length
        self.segment_length = segment_length

    def setup(self, stage: str = None):
        dfs = []
        for file in glob(f"{self.path}*.csv"):
            df = pd.read_csv(file)
            if len(df) < self.segment_length:
                logging.info(f"File {file} has length {len(df):,} instead of {self.segment_length}, skipping")
                continue
            if len(df) > self.segment_length:
                logging.info(f"File {file} has length {len(df)} instead of {self.segment_length}, truncating")
                df = df.iloc[:self.segment_length]
            df = df[self.x_cols + [self.y_col]]
            dfs.append(df)

        train_idxs = np.random.choice(len(dfs), int(0.8 * len(dfs)), replace=False)
        train_dfs = [dfs[i] for i in train_idxs]
        val_dfs = [df for i, df in enumerate(dfs) if i not in train_idxs]
        self.train = LatAccelDataset(train_dfs, self.input_length, self.x_cols, self.y_col, self.segment_length)
        print(f"Training set has {len(self.train)} samples")
        self.val = LatAccelDataset(val_dfs, self.input_length, self.x_cols, self.y_col, self.segment_length)
        print(f"Validation set has {len(self.val)} samples")

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
