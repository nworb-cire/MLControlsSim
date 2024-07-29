from data import LatAccelDataModule
from model import MLControlsSim
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger


def validate():
    model = MLControlsSim.load_from_checkpoint("lightning_logs/version_9/checkpoints/epoch=0-step=1000.ckpt")
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/000",
        batch_size=2 ** 10,
    )
    trainer = pl.Trainer()
    trainer.validate(model, datamodule=data)


def main():
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/",
        batch_size=2 ** 10,
    )
    model = MLControlsSim(
        n_layers=4,
        n_head=4,
        n_embd=128,
        lr=6e-4,
        weight_decay=0.1,
    )
    logger = CSVLogger(".")
    trainer = pl.Trainer(
        max_steps=10_000,
        precision=32,
        logger=logger,
        val_check_interval=200,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
