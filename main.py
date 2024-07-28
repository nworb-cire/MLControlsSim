from data import LatAccelDataModule
from model import MLControlsSim
import pytorch_lightning as pl


def main():
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/000",
        batch_size=2 ** 10,
    )
    model = MLControlsSim(
        n_layers=3,
        n_head=3,
        n_embd=48,
        lr=6e-4,
        weight_decay=0.1,
    )
    trainer = pl.Trainer(
        max_epochs=20,
        precision=32,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
