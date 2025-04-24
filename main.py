import argparse
import wandb
import json
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split

from src.models.til_dataset import TILDataset
from src.models.model_factory import SegModel


def main():
    print("Starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--mask_dir_name", type=str, default="raw_otsu")
    parser.add_argument("--wandb", type=str, default="")
    parser.add_argument("--config", type=str, default="./configs/models/base.yaml")
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--encoder_weights", type=str, default="imagenet")
    args = parser.parse_args()

    # Load hyperparameters from the YAML config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]

    wandb.login(key=args.wandb)

    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath="outputs/checkpoints/",
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=5, mode="min"
    )

    wandb_logger = WandbLogger(
        project=f"segmentation_{args.model_name}",
        name=args.mask_dir_name,
        save_dir="outputs/",
    )
    wandb_logger.log_hyperparams(vars(args))

    # Initialize the full training dataset
    full_train_dataset = TILDataset(
        os.path.join(args.data_path, "train/images"),
        os.path.join(args.data_path, "train/masks", str(args.mask_dir_name))
    )

    # Split into training and validation datasets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    pl.seed_everything(42)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Test dataset
    test_dataset = TILDataset(
        os.path.join(args.data_path, "test/images"),
        os.path.join(args.data_path, "test/masks")
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Training
    model = SegModel(
        model_name=args.model_name,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        learning_rate=learning_rate
    )

    trainer = pl.Trainer(
        default_root_dir="outputs/",
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop],
    )

    trainer.fit(model, train_loader, val_loader)

    # Evaluation
    best_model_path = checkpoint_callback.best_model_path
    best_model = SegModel.load_from_checkpoint(best_model_path)
    test_results = trainer.test(best_model, dataloaders=test_loader)

    with open("outputs/test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)

    wandb.finish()
    print("The End.")


if __name__ == "__main__":
    main()
