import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, random_split

from src.models.til_dataset import TILDataset
from src.models.unet import UNet


checkpoint_callback = ModelCheckpoint(
    dirpath="outputs/checkpoints/",
    filename="best-checkpoint",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
)

wandb_logger = WandbLogger(
    project="segmentation_project",
    name="unet_resnet34",
    save_dir="outputs/wandb/",
)

# Initialize the full training dataset
full_train_dataset = TILDataset("./data/processed-train/patches/images",
                                "./data/processed-train/patches/masks/raw_otsu")

# Split into training and validation datasets
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Test dataset
test_dataset = TILDataset("./data/processed-test/patches/images",
                          "./data/processed-test/patches/masks")
test_loader = DataLoader(test_dataset, batch_size=16)

# Training
model = UNet()

trainer = pl.Trainer(
    default_root_dir="outputs/",
    max_epochs=10,
    logger=wandb_logger,
    log_every_n_steps=10,
    accelerator="auto",
    callbacks=[checkpoint_callback],
)

trainer.fit(model, train_loader, val_loader)

best_model_path = checkpoint_callback.best_model_path
best_model = UNet.load_from_checkpoint(best_model_path)
trainer.test(best_model, dataloaders=test_loader)
