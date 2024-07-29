from glob import glob
import logging

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import LataccelTokenizer


class LatAccelDataset(Dataset):
    def __init__(self, data, segment_length: int, context_size: int):
        assert data.ndim == 3
        self.data = torch.tensor(data, dtype=torch.float32)
        self.segment_length = segment_length
        self.context_size = context_size

    def __len__(self):
        return self.data.shape[0] * (self.segment_length - 2 * self.context_size)

    def __getitem__(self, idx):
        seq_idx = idx // (self.segment_length - 2 * self.context_size)
        start_idx = idx % (self.segment_length - 2 * self.context_size)
        end_idx = start_idx + 2 * self.context_size

        x = self.data[seq_idx, start_idx:end_idx, :-2]
        y = self.data[seq_idx, start_idx:end_idx, -2].to(dtype=torch.long)
        mask = self.data[seq_idx, start_idx:end_idx, -1].to(dtype=torch.bool)
        return x, y, mask


class LatAccelDataModule(pl.LightningDataModule):
    x_cols = [
        "steerFiltered",
        "roll",
        "vEgo",
        "aEgo",
        "latAccelLocalizer",
    ]
    y_col = "latAccelLocalizer"
    SEGMENT_LENGTH = 512
    CONTEXT_SIZE = 20

    def __init__(self, path: str, batch_size: int):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.tokenizer = LataccelTokenizer()

    def setup(self, stage: str = None):
        segments = []
        for file in glob(f"{self.path}*.csv"):
            df = pd.read_csv(file)
            if len(df) < self.SEGMENT_LENGTH:
                logging.warning(f"File {file} is too short: {len(df)}")
                continue
            df = df[self.x_cols + [self.y_col, "latActive", "steeringPressed"]]
            df["roll"] = np.sin(df["roll"]) * 9.81
            df[self.y_col] = self.tokenizer.encode(df[self.y_col])
            df["mask"] = ~df["latActive"] | df["steeringPressed"]
            # mask before and after 10 time steps
            df["mask"] = df["mask"].rolling(self.CONTEXT_SIZE, center=True, min_periods=1).max()
            df["mask"] = df["mask"].shift(-5).fillna(1.0)
            df.drop(columns=["latActive", "steeringPressed"], inplace=True)
            df = df.iloc[:self.SEGMENT_LENGTH]
            if df["mask"].any():  # TODO: include more data, just not where it's masked for too long
                # logging.warning(f"File {file} contains no valid data")
                continue
            # add batch dimension
            val = df.values[np.newaxis]
            segments.append(val)

        # Concatenate: (n_sequences, sequence_length, n_features)
        data = np.concatenate(segments, axis=0)

        train_size = int(data.shape[0] * 0.9)
        print(f"Train size: {train_size:,}, Val size: {data.shape[0] - train_size:,}")
        self.train = LatAccelDataset(data[:train_size, :, :], self.SEGMENT_LENGTH, self.CONTEXT_SIZE)
        self.val = LatAccelDataset(data[train_size:, :, :], self.SEGMENT_LENGTH, self.CONTEXT_SIZE)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, persistent_workers=True)
