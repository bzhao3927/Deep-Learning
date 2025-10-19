import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

class UNetPlusPlusModule(pl.LightningModule):
    def __init__(self, classes=3, lr=1e-3, weight_decay=1e-4, encoder_name="resnet34", use_dice_loss=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Use smp UNet++ with configurable encoder
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=classes,
            activation=None
        )
        
        # Loss function: CrossEntropy or Dice+CE combo
        if use_dice_loss:
            # Combo loss: 50% Dice + 50% CrossEntropy
            self.dice_loss = DiceLoss(mode='multiclass')
            self.ce_loss = nn.CrossEntropyLoss()
            self.loss_fn = lambda logits, targets: 0.5 * self.dice_loss(logits, targets) + 0.5 * self.ce_loss(logits, targets)
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.classes = classes

        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(
            num_classes=classes,
            task="multiclass"
        )
        self.val_iou = torchmetrics.JaccardIndex(
            num_classes=classes,
            task="multiclass"
        )
        self.val_dice = torchmetrics.F1Score(
            num_classes=classes,
            task="multiclass",
            average="macro"
        )

    def forward(self, x):
        return self.model(x)
    
    def _initialize_decoder_weights(self):
        """Initialize decoder weights using Kaiming initialization."""
        for name, module in self.model.decoder.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        for module in self.model.segmentation_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def step(self, batch):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.train_iou(preds, masks)
        self.log("train_mIoU", self.train_iou, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        self.log("val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        self.log("test_loss", loss)
        self.log("test_mIoU", self.val_iou(preds, masks))
        self.log("test_Dice", self.val_dice(preds, masks))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }