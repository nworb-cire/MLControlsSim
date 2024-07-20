from data import LatAccelDataModule
from model import MLControlsSim
import pytorch_lightning as pl


def main():
    model = MLControlsSim(
        input_dim=4,
        d_model=128,
        nhead=4,
        num_layers=4,
        input_length=10,
    )
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/",
        x_cols=["steerFiltered", "roll", "vEgo", "aEgo"],
        y_col="latAccelLocalizer",
        batch_size=2 ** 10,
        input_length=10,
    )
    trainer = pl.Trainer(
        max_epochs=20,
        precision=32,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
