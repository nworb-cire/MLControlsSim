import pytorch_lightning as pl
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, input_length):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc = nn.Linear(d_model, 2 * input_length)

    def forward(self, src):
        src = self.embedding(src)
        src = src.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, d_model)
        transformer_output = self.transformer(src, src)
        output = self.fc(transformer_output[-1])  # We take the output of the last time step
        return output.squeeze()


class MLControlsSim(pl.LightningModule):
    def __init__(self, input_dim, d_model, nhead, num_layers, input_length, lr=1e-4):
        super().__init__()
        self.model = TransformerModel(input_dim, d_model, nhead, num_layers, input_length)
        self.lr = lr

    def criterion(self, y_hat, y):
        """NLL of Laplace distribution"""
        mu = y_hat[:, 0::2]
        mu = torch.cumsum(mu, dim=1)
        theta = y_hat[:, 1::2]
        theta = torch.cumsum(theta, dim=1)
        loss = torch.mean((torch.abs(mu - y) / torch.exp(theta)) + theta)
        return loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
