"""
src/model_unetpp.py - UNet++ with CBAM Attention Modules
Improvements: Hybrid Loss + Deep Supervision + TTA + Channel & Spatial Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


# ============================================================================
# Attention Modules
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) - focuses on 'what' is meaningful"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP for both pooling branches
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        
        # Apply global pooling and shared MLP
        avg_pool_out = self.shared_mlp(self.avg_pool(x).view(batch_size, num_channels))
        max_pool_out = self.shared_mlp(self.max_pool(x).view(batch_size, num_channels))
        
        # Combine and create attention map
        attention_map = self.sigmoid(avg_pool_out + max_pool_out).view(batch_size, num_channels, 1, 1)
        
        return x * attention_map


class SpatialAttention(nn.Module):
    """Spatial Attention Module (SAM) - focuses on 'where' is meaningful"""
    
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise pooling
        avg_channel_pool = torch.mean(x, dim=1, keepdim=True)
        max_channel_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and generate attention map
        pooled_features = torch.cat([avg_channel_pool, max_channel_pool], dim=1)
        attention_map = self.sigmoid(self.conv(pooled_features))
        
        return x * attention_map


class CBAM(nn.Module):
    """Convolutional Block Attention Module - combines Channel and Spatial Attention"""
    
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance by down-weighting easy examples"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        prob_true = torch.exp(-ce_loss)
        focal_loss = (1 - prob_true) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for direct IoU optimization in segmentation tasks"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        
        return dice_loss.mean()


class HybridLoss(nn.Module):
    """Combines Focal Loss (for classification) and Dice Loss (for segmentation)"""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, 
                 focal_gamma=2.0, class_weights=None):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.focal_weight * focal + self.dice_weight * dice


# ============================================================================
# UNet++ Building Blocks
# ============================================================================

class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm, ReLU, and optional CBAM attention"""
    
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = CBAM(out_channels) if use_attention else None
    
    def forward(self, x):
        x = self.conv_layers(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


# ============================================================================
# UNet++ Architecture
# ============================================================================

class UNetPlusPlus(nn.Module):
    """
    UNet++ with nested skip pathways, configurable depth, and optional attention
    
    Args:
        in_channels: Number of input channels (e.g., 3 for RGB images)
        num_classes: Number of output segmentation classes
        base_channels: Base number of feature channels (doubled at each depth level)
        depth: Network depth (4, 5, or 6 levels)
        deep_supervision: If True, outputs predictions at multiple depths
        use_attention: If True, applies CBAM attention in decoder blocks
    """
    
    def __init__(self, in_channels=3, num_classes=3, base_channels=32, depth=4, 
                 deep_supervision=True, use_attention=False):
        super().__init__()
        
        self.depth = depth
        self.deep_supervision = deep_supervision
        
        # Calculate channel dimensions for each level
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # ====================================================================
        # Encoder Path (Contracting - no attention)
        # ====================================================================
        self.pool = nn.MaxPool2d(2, 2)
        
        # Level 0 (highest resolution)
        self.encoder_0_0 = ConvBlock(in_channels, channels[0], use_attention=False)
        self.encoder_1_0 = ConvBlock(channels[0], channels[1], use_attention=False)
        self.encoder_2_0 = ConvBlock(channels[1], channels[2], use_attention=False)
        self.encoder_3_0 = ConvBlock(channels[2], channels[3], use_attention=False)
        self.encoder_4_0 = ConvBlock(channels[3], channels[4], use_attention=False)
        
        # Additional depth levels
        if depth >= 5:
            self.encoder_5_0 = ConvBlock(channels[4], channels[5], use_attention=False)
        if depth >= 6:
            self.encoder_6_0 = ConvBlock(channels[5], channels[6], use_attention=False)
        
        # ====================================================================
        # Decoder Path with Nested Skip Connections (with optional attention)
        # ====================================================================
        
        # Column 1 (first skip connection)
        self.decoder_0_1 = ConvBlock(channels[0] + channels[1], channels[0], use_attention=use_attention)
        self.decoder_1_1 = ConvBlock(channels[1] + channels[2], channels[1], use_attention=use_attention)
        self.decoder_2_1 = ConvBlock(channels[2] + channels[3], channels[2], use_attention=use_attention)
        self.decoder_3_1 = ConvBlock(channels[3] + channels[4], channels[3], use_attention=use_attention)
        
        # Column 2 (second skip connection)
        self.decoder_0_2 = ConvBlock(channels[0] * 2 + channels[1], channels[0], use_attention=use_attention)
        self.decoder_1_2 = ConvBlock(channels[1] * 2 + channels[2], channels[1], use_attention=use_attention)
        self.decoder_2_2 = ConvBlock(channels[2] * 2 + channels[3], channels[2], use_attention=use_attention)
        
        # Column 3 (third skip connection)
        self.decoder_0_3 = ConvBlock(channels[0] * 3 + channels[1], channels[0], use_attention=use_attention)
        self.decoder_1_3 = ConvBlock(channels[1] * 3 + channels[2], channels[1], use_attention=use_attention)
        
        # Column 4 (fourth skip connection)
        self.decoder_0_4 = ConvBlock(channels[0] * 4 + channels[1], channels[0], use_attention=use_attention)
        
        # Additional columns for deeper networks
        if depth >= 5:
            self.decoder_4_1 = ConvBlock(channels[4] + channels[5], channels[4], use_attention=use_attention)
            self.decoder_3_2 = ConvBlock(channels[3] * 2 + channels[4], channels[3], use_attention=use_attention)
            self.decoder_2_3 = ConvBlock(channels[2] * 3 + channels[3], channels[2], use_attention=use_attention)
            self.decoder_1_4 = ConvBlock(channels[1] * 4 + channels[2], channels[1], use_attention=use_attention)
            self.decoder_0_5 = ConvBlock(channels[0] * 5 + channels[1], channels[0], use_attention=use_attention)
        
        if depth >= 6:
            self.decoder_5_1 = ConvBlock(channels[5] + channels[6], channels[5], use_attention=use_attention)
            self.decoder_4_2 = ConvBlock(channels[4] * 2 + channels[5], channels[4], use_attention=use_attention)
            self.decoder_3_3 = ConvBlock(channels[3] * 3 + channels[4], channels[3], use_attention=use_attention)
            self.decoder_2_4 = ConvBlock(channels[2] * 4 + channels[3], channels[2], use_attention=use_attention)
            self.decoder_1_5 = ConvBlock(channels[1] * 5 + channels[2], channels[1], use_attention=use_attention)
            self.decoder_0_6 = ConvBlock(channels[0] * 6 + channels[1], channels[0], use_attention=use_attention)
        
        # ====================================================================
        # Final Output Layers (1x1 convolutions)
        # ====================================================================
        if deep_supervision:
            self.output_1 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            self.output_2 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            self.output_3 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            self.output_4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            if depth >= 5:
                self.output_5 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
            if depth >= 6:
                self.output_6 = nn.Conv2d(channels[0], num_classes, kernel_size=1)
        else:
            self.output_final = nn.Conv2d(channels[0], num_classes, kernel_size=1)
    
    def _upsample(self, x, target_size):
        """Helper method for upsampling with bilinear interpolation"""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # ====================================================================
        # Encoder Forward Pass
        # ====================================================================
        x0_0 = self.encoder_0_0(x)
        x1_0 = self.encoder_1_0(self.pool(x0_0))
        x2_0 = self.encoder_2_0(self.pool(x1_0))
        x3_0 = self.encoder_3_0(self.pool(x2_0))
        x4_0 = self.encoder_4_0(self.pool(x3_0))
        
        if self.depth >= 5:
            x5_0 = self.encoder_5_0(self.pool(x4_0))
        if self.depth >= 6:
            x6_0 = self.encoder_6_0(self.pool(x5_0))
        
        # ====================================================================
        # Decoder with Nested Skip Connections
        # ====================================================================
        target_size = x0_0.shape[2:]
        
        # Column 1
        x0_1 = self.decoder_0_1(torch.cat([x0_0, self._upsample(x1_0, target_size)], dim=1))
        x1_1 = self.decoder_1_1(torch.cat([x1_0, self._upsample(x2_0, x1_0.shape[2:])], dim=1))
        x2_1 = self.decoder_2_1(torch.cat([x2_0, self._upsample(x3_0, x2_0.shape[2:])], dim=1))
        x3_1 = self.decoder_3_1(torch.cat([x3_0, self._upsample(x4_0, x3_0.shape[2:])], dim=1))
        
        # Column 2
        x0_2 = self.decoder_0_2(torch.cat([x0_0, x0_1, self._upsample(x1_1, target_size)], dim=1))
        x1_2 = self.decoder_1_2(torch.cat([x1_0, x1_1, self._upsample(x2_1, x1_0.shape[2:])], dim=1))
        x2_2 = self.decoder_2_2(torch.cat([x2_0, x2_1, self._upsample(x3_1, x2_0.shape[2:])], dim=1))
        
        # Column 3
        x0_3 = self.decoder_0_3(torch.cat([x0_0, x0_1, x0_2, self._upsample(x1_2, target_size)], dim=1))
        x1_3 = self.decoder_1_3(torch.cat([x1_0, x1_1, x1_2, self._upsample(x2_2, x1_0.shape[2:])], dim=1))
        
        # Column 4
        x0_4 = self.decoder_0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._upsample(x1_3, target_size)], dim=1))
        
        # Additional columns for deeper networks
        if self.depth >= 5:
            x4_1 = self.decoder_4_1(torch.cat([x4_0, self._upsample(x5_0, x4_0.shape[2:])], dim=1))
            x3_2 = self.decoder_3_2(torch.cat([x3_0, x3_1, self._upsample(x4_1, x3_0.shape[2:])], dim=1))
            x2_3 = self.decoder_2_3(torch.cat([x2_0, x2_1, x2_2, self._upsample(x3_2, x2_0.shape[2:])], dim=1))
            x1_4 = self.decoder_1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self._upsample(x2_3, x1_0.shape[2:])], dim=1))
            x0_5 = self.decoder_0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self._upsample(x1_4, target_size)], dim=1))
        
        if self.depth >= 6:
            x5_1 = self.decoder_5_1(torch.cat([x5_0, self._upsample(x6_0, x5_0.shape[2:])], dim=1))
            x4_2 = self.decoder_4_2(torch.cat([x4_0, x4_1, self._upsample(x5_1, x4_0.shape[2:])], dim=1))
            x3_3 = self.decoder_3_3(torch.cat([x3_0, x3_1, x3_2, self._upsample(x4_2, x3_0.shape[2:])], dim=1))
            x2_4 = self.decoder_2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self._upsample(x3_3, x2_0.shape[2:])], dim=1))
            x1_5 = self.decoder_1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self._upsample(x2_4, x1_0.shape[2:])], dim=1))
            x0_6 = self.decoder_0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self._upsample(x1_5, target_size)], dim=1))
        
        # ====================================================================
        # Output
        # ====================================================================
        if self.deep_supervision:
            outputs = [
                self.output_1(x0_1),
                self.output_2(x0_2),
                self.output_3(x0_3),
                self.output_4(x0_4)
            ]
            if self.depth >= 5:
                outputs.append(self.output_5(x0_5))
            if self.depth >= 6:
                outputs.append(self.output_6(x0_6))
            return outputs
        else:
            # Return the deepest output
            final_features = {4: x0_4, 5: x0_5, 6: x0_6}[self.depth]
            return self.output_final(final_features)


# ============================================================================
# PyTorch Lightning Module
# ============================================================================

class UNetPlusPlusModule(pl.LightningModule):
    """PyTorch Lightning wrapper for UNet++ with training, validation, and testing logic"""
    
    def __init__(self, in_channels=3, num_classes=3, base_channels=32, depth=4,
                 deep_supervision=True, use_attention=False, lr=1e-3, weight_decay=1e-4, 
                 max_epochs=50, use_hybrid_loss=False, use_tta=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = UNetPlusPlus(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            depth=depth,
            deep_supervision=deep_supervision,
            use_attention=use_attention
        )
        
        # Loss function
        if use_hybrid_loss:
            self.criterion = HybridLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass")
        self.val_iou = torchmetrics.JaccardIndex(num_classes=num_classes, task="multiclass")
        self.val_dice = torchmetrics.F1Score(num_classes=num_classes, task="multiclass", average="macro")
    
    def forward(self, x):
        return self.model(x)
    
    def _apply_tta(self, images):
        """Test-Time Augmentation: Average predictions over 8 transformations"""
        def get_final_logits(model_output):
            """Extract final logits from model output (handles deep supervision)"""
            return model_output[-1] if isinstance(model_output, list) else model_output
        
        # Original prediction
        pred = torch.softmax(get_final_logits(self(images)), dim=1)
        
        # Horizontal flip
        images_hflip = torch.flip(images, dims=[3])
        pred += torch.flip(torch.softmax(get_final_logits(self(images_hflip)), dim=1), dims=[3])
        
        # Vertical flip
        images_vflip = torch.flip(images, dims=[2])
        pred += torch.flip(torch.softmax(get_final_logits(self(images_vflip)), dim=1), dims=[2])
        
        # Both flips
        images_hvflip = torch.flip(images, dims=[2, 3])
        pred += torch.flip(torch.softmax(get_final_logits(self(images_hvflip)), dim=1), dims=[2, 3])
        
        # Rotations: 90°, 180°, 270°
        for k in [1, 2, 3]:
            images_rot = torch.rot90(images, k=k, dims=[2, 3])
            pred_rot = torch.softmax(get_final_logits(self(images_rot)), dim=1)
            pred += torch.rot90(pred_rot, k=-k, dims=[2, 3])
        
        # 90° rotation + horizontal flip
        images_rot90_hflip = torch.flip(torch.rot90(images, k=1, dims=[2, 3]), dims=[3])
        pred_rot90_hflip = torch.softmax(get_final_logits(self(images_rot90_hflip)), dim=1)
        pred += torch.rot90(torch.flip(pred_rot90_hflip, dims=[3]), k=-1, dims=[2, 3])
        
        return pred / 8.0
    
    def _compute_loss_and_predictions(self, batch):
        """Shared logic for computing loss and predictions"""
        images, masks = batch
        model_outputs = self(images)
        
        if self.hparams.deep_supervision:
            # Average loss across all supervision outputs
            loss = sum(self.criterion(output, masks) for output in model_outputs) / len(model_outputs)
            logits = model_outputs[-1]  # Use deepest output for metrics
        else:
            logits = model_outputs
            loss = self.criterion(logits, masks)
        
        predictions = torch.argmax(logits, dim=1)
        return loss, predictions, masks
    
    def training_step(self, batch, batch_idx):
        loss, predictions, masks = self._compute_loss_and_predictions(batch)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_iou.update(predictions, masks)
        self.log("train_mIoU", self.train_iou, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, predictions, masks = self._compute_loss_and_predictions(batch)
        
        self.log("val_loss", loss, prog_bar=True)
        self.val_iou.update(predictions, masks)
        self.val_dice.update(predictions, masks)
        self.log("val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, predictions, masks = self._compute_loss_and_predictions(batch)
        
        self.log("test_loss", loss)
        self.log("test_mIoU", self.val_iou(predictions, masks))
        self.log("test_Dice", self.val_dice(predictions, masks))
    
    def predict_step(self, batch, batch_idx):
        """Prediction with optional TTA"""
        images = batch if isinstance(batch, torch.Tensor) else batch[0]
        
        if self.hparams.use_tta:
            return self._apply_tta(images)
        else:
            outputs = self(images)
            logits = outputs[-1] if isinstance(outputs, list) else outputs
            return torch.softmax(logits, dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }