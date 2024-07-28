from data import LatAccelDataModule
from model import MLControlsSim
import pytorch_lightning as pl


def main():
    data = LatAccelDataModule(
        path="../NNFF/data/CHEVROLET_VOLT_PREMIER_2017/0",
        batch_size=2 ** 10,
    )
    model = MLControlsSim(
        input_dim=len(data.x_cols),
        d_model=48,
        nhead=3,
        num_layers=3,
        input_length=10,
    )
    trainer = pl.Trainer(
        max_epochs=20,
        precision=32,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
