from data import LatAccelDataModule
from model import MLControlsSim
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger


def main():
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/",
        batch_size=2 ** 7,
    )
    model = MLControlsSim(
        n_layers=3,
        n_head=4,
        n_embd=64,
        lr=6e-4,
        weight_decay=0.1,
    )
    logger = CSVLogger(".")
    trainer = pl.Trainer(
        max_epochs=500,
        precision=32,
        overfit_batches=3,
        logger=logger,
        val_check_interval=10,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
