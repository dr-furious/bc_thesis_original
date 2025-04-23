import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import get_stats
from segmentation_models_pytorch.metrics.functional import iou_score, f1_score
import torch.nn as nn
import torch


class UNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.sigmoid(logits) > 0.5
        tp, fp, fn, tn = get_stats(preds.long(), masks.long(), mode='binary')

        iou = iou_score(tp, fp, fn, tn, reduction='micro')
        f1 = f1_score(tp, fp, fn, tn, reduction='micro')

        self.log("val_loss", loss, on_epoch=True)
        self.log("val_iou", iou, on_epoch=True)
        self.log("val_f1", f1, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        preds = torch.sigmoid(logits) > 0.5
        tp, fp, fn, tn = get_stats(preds.long(), masks.long(), mode='binary')

        iou = iou_score(tp, fp, fn, tn, reduction='micro')
        f1 = f1_score(tp, fp, fn, tn, reduction='micro')

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_iou", iou, on_epoch=True)
        self.log("test_f1", f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
