import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import wandb
from segmentation_models_pytorch.metrics import get_stats
from segmentation_models_pytorch.metrics.functional import iou_score, f1_score, accuracy, recall, precision
import torch


class SegModel(pl.LightningModule):
    def __init__(
        self,
        model_instance=None,
        model_name: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: str = None,
        in_channels: int = 3,
        classes: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        if model_instance is None:
            # Get the model class from smp
            try:
                model_class = getattr(smp, model_name)
            except AttributeError:
                raise ValueError(f"Model {model_name} is not available in segmentation_models_pytorch.")
            # Create the model
            self.model = model_class(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
            )
        else:
            self.model = model_instance

        # Set loss function of the model
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    # This shared step is used both for training, validation, and testing
    def _shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        # Compute true postives, false positives, false negatives and true negatives, pixel-wise
        tp, fp, fn, tn = get_stats(logits, masks.long(), mode="binary", threshold=0.5)

        # Compute evaluation metrics
        iou = iou_score(tp, fp, fn, tn, reduction="micro", zero_division=0.0)
        f1 = f1_score(tp, fp, fn, tn, reduction="micro", zero_division=0.0)
        acc = accuracy(tp, fp, fn, tn, reduction="micro", zero_division=0.0)
        pre = precision(tp, fp, fn, tn, reduction="micro", zero_division=0.0)
        rec = recall(tp, fp, fn, tn, reduction="micro", zero_division=0.0)

        # Log all metrics to wandb via Logger
        self.log(f"{stage}/loss", loss, on_epoch=True)
        self.log(f"{stage}/iou", iou, on_epoch=True)
        self.log(f"{stage}/f1", f1, on_epoch=True)
        self.log(f"{stage}/accuracy", acc, on_epoch=True)
        self.log(f"{stage}/precision", pre, on_epoch=True)
        self.log(f"{stage}/recall", rec, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "val")
        if batch_idx % 10 == 0:
            self._log_images(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "test")
        if batch_idx % 10 == 0:
            self._log_images(batch, "test")
        return loss

    def _log_images(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Pull image, mask, and preds to cpu and convert them to numpy arrays
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()
        preds = preds.cpu().numpy()

        # Log max 6 images (for wandb performance)
        num_images = min(6, images.shape[0])
        logged_images = []
        for i in range(num_images):
            # Convert from CHW to HWC
            img = images[i].transpose(1, 2, 0)
            # Masks with shape (B, 1, H, W)
            mask = masks[i][0]
            pred = preds[i][0]

            # Convert to uint8
            img = img.astype(np.uint8)
            mask = mask.astype(np.uint8)
            pred = pred.astype(np.uint8)

            # Create a wandb.Image with masks
            wandb_image = wandb.Image(
                img,
                masks={
                    "ground_truth": {
                        "mask_data": mask,
                        "class_labels": {0: "background", 1: "lymphocyte"},
                    },
                    "prediction": {
                        "mask_data": pred,
                        "class_labels": {0: "background", 1: "lymphocyte"},
                    },
                },
            )
            logged_images.append(wandb_image)

        # The below warning is OK as long as WandbLogger is used with the Trainer
        self.logger.experiment.log({f"{stage}/examples": logged_images})

    def configure_optimizers(self):
        trainable_p = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(
            trainable_p,
            lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]

