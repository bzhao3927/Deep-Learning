import argparse
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from tqdm import tqdm

from datamodule_oxpet import OxfordPetDataModule
from model_unetpp import UNetPlusPlusModule
from viz import create_comparison_grid, save_single_prediction
import torchmetrics


def train(args):
    """Train the model"""
    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Initialize data module
    dm = OxfordPetDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        classes=args.classes,
        use_augmentation=args.use_augmentation  # NEW
    )
    
    # Initialize model
    num_classes = 3 if args.classes == 'trimap' else 2
    model = UNetPlusPlusModule(
        classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_name=args.encoder,
        use_dice_loss=args.use_dice_loss  # NEW
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir / 'checkpoints',
        filename='unetpp-{epoch:02d}-{val_mIoU:.4f}',
        monitor='val_mIoU',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.exp_name,
            save_dir=args.output_dir / 'logs'
        )
    else:
        logger = CSVLogger(
            save_dir=args.output_dir / 'logs',
            name=args.exp_name
        )
    
    # Trainer
    if torch.cuda.is_available() and torch.version.cuda.startswith("12"):
        accelerator = 'gpu'
        devices = 1
        precision = args.precision
    else:
        print("WARNING: CUDA not available or incompatible. Using CPU.")
        accelerator = 'cpu'
        devices = 1
        precision = 32  # AMP only works on GPU

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        deterministic=False
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.fit(model, dm)
    
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_mIoU: {checkpoint_callback.best_model_score:.4f}")
    
    return checkpoint_callback.best_model_path


def evaluate(checkpoint_path, args):
    """Evaluate model on test set and generate visualizations"""
    print("\n" + "="*60)
    print("Starting evaluation on test set...")
    print("="*60)
    
    # Load data
    dm = OxfordPetDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        classes=args.classes,
        use_augmentation=False  # Don't augment during eval
    )
    dm.setup()
    
    # Load best model
    model = UNetPlusPlusModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test dataloader
    test_loader = dm.test_dataloader()
    
    # Metrics
    num_classes = 3 if args.classes == 'trimap' else 2
    iou_metric = torchmetrics.JaccardIndex(
        num_classes=num_classes, 
        task="multiclass",
        average='none'
    ).to(device)
    
    dice_metric = torchmetrics.F1Score(
        num_classes=num_classes,
        task="multiclass",
        average='none'
    ).to(device)
    
    # Collect samples for visualization
    vis_images = []
    vis_gt_masks = []
    vis_pred_masks = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            
            # Update metrics
            iou_metric.update(preds, masks)
            dice_metric.update(preds, masks)
            
            # Collect samples for visualization (at least 12)
            if len(vis_images) < 12:
                for i in range(imgs.shape[0]):
                    if len(vis_images) >= 12:
                        break
                    vis_images.append(imgs[i])
                    vis_gt_masks.append(masks[i])
                    vis_pred_masks.append(preds[i])
    
    # Compute final metrics
    iou_per_class = iou_metric.compute().cpu().numpy()
    dice_per_class = dice_metric.compute().cpu().numpy()
    
    mean_iou = iou_per_class.mean()
    mean_dice = dice_per_class.mean()
    
    # Print results
    class_names = ['Pet', 'Background', 'Border'] if args.classes == 'trimap' else ['Background', 'Pet']
    
    print("\n" + "="*60)
    print("Test Set Results:")
    print("="*60)
    print(f"\nOverall Metrics:")
    print(f"  Mean IoU:  {mean_iou:.4f}")
    print(f"  Mean Dice: {mean_dice:.4f}")
    
    print(f"\nPer-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {iou_per_class[i]:.4f}")
    
    print(f"\nPer-Class Dice:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {dice_per_class[i]:.4f}")
    print("="*60 + "\n")
    
    # Create visualizations
    samples_dir = args.output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating visualizations with {len(vis_images)} samples...")
    
    # Create comparison grid
    create_comparison_grid(
        images=vis_images,
        gt_masks=vis_gt_masks,
        pred_masks=vis_pred_masks,
        mode=args.classes,
        denorm=True,
        save_path=samples_dir / 'test_comparison_grid.png'
    )
    
    # Save individual samples
    for i in range(len(vis_images)):
        save_single_prediction(
            image=vis_images[i],
            gt_mask=vis_gt_masks[i],
            pred_mask=vis_pred_masks[i],
            save_path=samples_dir / f'sample_{i:03d}.png',
            mode=args.classes,
            denorm=True
        )
    
    print(f"Visualizations saved to {samples_dir}")
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'iou_per_class': iou_per_class,
        'dice_per_class': dice_per_class
    }


def main():
    parser = argparse.ArgumentParser(description='Train UNet++ on Oxford-IIIT Pet')
    
    # Data
    parser.add_argument('--data_root', type=str, default='~/data/oxford-iiit-pet',
                        help='Path to Oxford-IIIT Pet dataset')
    parser.add_argument('--classes', type=str, default='trimap', choices=['trimap', 'binary'],
                        help='Classification mode')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34',
                        choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet-b0', 
                                 'efficientnet-b3', 'mobilenet_v2'],
                        help='Encoder backbone')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision (AMP)')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--exp_name', type=str, default='baseline',
                        help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='oxpet-segmentation',
                        help='W&B project name')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'both'],
                        help='Run mode: train only, eval only, or both')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path for evaluation (if mode=eval)')

    # NEW options
    parser.add_argument('--use_dice_loss', action='store_true',
                        help='Use Dice+CE combo loss instead of just CE')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use data augmentation for training')
    
    args = parser.parse_args()
    
    # Convert paths
    args.data_root = Path(args.data_root).expanduser()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run based on mode
    if args.mode == 'train':
        best_ckpt = train(args)
        print(f"\nTo evaluate this model, run:")
        print(f"python src/train.py --mode eval --checkpoint {best_ckpt}")
        
    elif args.mode == 'eval':
        if args.checkpoint is None:
            print("Error: --checkpoint required for evaluation mode")
            return
        evaluate(args.checkpoint, args)
        
    elif args.mode == 'both':
        best_ckpt = train(args)
        evaluate(best_ckpt, args)


if __name__ == '__main__':
    main()
