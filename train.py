import datasets
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter, Compose, ToTensor


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    model = MNISTModel()

    ds = datasets.load_dataset("mnist")

    loaders = {k: torch.utils.data.DataLoader(v, batch_size=32) for k, v in ds.items()}

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        max_epochs=3,
        callbacks=[pl.callbacks.progress.TQDMProgressBar(refresh_rate=20)],
    )

    trainer.fit(model, loaders["train"])
