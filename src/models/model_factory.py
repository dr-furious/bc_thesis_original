import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.metrics import get_stats
from segmentation_models_pytorch.metrics.functional import iou_score, f1_score, accuracy, recall, precision
import torch


class SegModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "Unet",
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Get the model class from smp
        try:
            model_class = getattr(smp, model_name)
        except AttributeError:
            raise ValueError(f"Model {model_name} is not available in segmentation_models_pytorch.")

        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        tp, fp, fn, tn = get_stats(logits, masks.long(), mode="binary", threshold=0.5)

        iou = iou_score(tp, fp, fn, tn, reduction="micro")
        f1 = f1_score(tp, fp, fn, tn, reduction="micro")
        acc = accuracy(tp, fp, fn, tn, reduction="micro")
        pre = precision(tp, fp, fn, tn, reduction="micro")
        rec = recall(tp, fp, fn, tn, reduction="micro")

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
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return [optimizer], [scheduler]
