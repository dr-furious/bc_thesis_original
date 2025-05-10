import argparse

import numpy as np
import torch
import wandb
import json
import yaml
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedGroupKFold

from src.models.til_dataset import TILDataset
from src.models.model_factory import SegModel


def main():
    print("Starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--train_images_path", type=str, default="train/images")
    parser.add_argument("--train_masks_path", type=str, default="train/masks")
    parser.add_argument("--test_images_path", type=str, default="test/images")
    parser.add_argument("--test_masks_path", type=str, default="test/masks")
    parser.add_argument("--wandb", type=str, default=None)
    parser.add_argument("--wandb_proj_name", type=str, default="segmentation")
    parser.add_argument("--wandb_run_name", type=str, default="run")
    parser.add_argument("--config", type=str, default="./configs/model_train_base.yaml")
    args = parser.parse_args()

    # List the dirs on the first level, just to check if they are alright
    print(os.listdir(args.data_path))
    # return

    # Load config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get config data
    max_epochs = config["max_epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    early_stop_patience = config["early_stop_patience"]
    model_name = config["model_name"]
    encoder_name = config["encoder_name"]
    encoder_weights = config["encoder_weights"]
    pretrained_ckpt = config["pretrained_ckpt"]
    freeze_encoder = config["freeze_encoder"]
    eval_only = config["eval_only"]
    split_token = config["split_token"]

    # If provided, login to wandb using the key, otherwise prompt user to do so
    if args.wandb is not None:
        wandb.login(key=args.wandb)
    else:
        wandb.login()

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
        patience=early_stop_patience,
        mode="min"
    )

    wandb_logger = WandbLogger(
        project=f"{args.wandb_proj_name}",
        name=args.wandb_run_name,
        save_dir="outputs/",
    )

    # Initialize the full training dataset
    full_train_dataset = TILDataset(
        os.path.join(args.data_path, str(args.train_images_path)),
        os.path.join(args.data_path, str(args.train_masks_path))
    )

    pl.seed_everything(42)

    # Extract group IDs from image IDs
    groups = [image_id.split(split_token, 1)[0] for image_id in full_train_dataset.image_ids]
    train_dataset = []
    val_dataset = []

    # Determine labels based on the presence of positive pixels in masks
    labels = []
    for idx in range(len(full_train_dataset)):
        _, mask = full_train_dataset[idx]
        labels.append(int(mask.sum() > 0))

    labels = np.array(labels)

    # Initialize StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform the split
    for train_idx, val_idx in sgkf.split(np.zeros(len(labels)), labels, groups):
        train_dataset = Subset(full_train_dataset, train_idx)
        val_dataset = Subset(full_train_dataset, val_idx)
        break  # Use only the first split

    # Train dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Test dataset
    test_dataset = TILDataset(
        os.path.join(args.data_path, str(args.test_images_path)),
        os.path.join(args.data_path, str(args.test_masks_path))
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # If using transfer learning, load the model from provided checkpoint:
    if pretrained_ckpt is not None:
        print(f"Loading pretrained model from {pretrained_ckpt}")
        model = SegModel.load_from_checkpoint(pretrained_ckpt)
    else:
        # Create a new model
        model = SegModel(
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            learning_rate=learning_rate
        )

    # Freeze the encoder
    if freeze_encoder is True and (encoder_weights is not None or pretrained_ckpt is not None):
        print("Freezing encoder...")
        for name, param in model.model.encoder.named_parameters():
            param.requires_grad = False
        print("Encoder is frozen.")

    # Training
    trainer = pl.Trainer(
        default_root_dir="outputs/",
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stop],
    )

    if eval_only is True:
        if pretrained_ckpt is None:
            raise ValueError("Please provide pretrained model.")
        print("Evaluating model...")
        model = SegModel.load_from_checkpoint(pretrained_ckpt)
        test_results = trainer.test(model, dataloaders=test_loader)
    else:
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
