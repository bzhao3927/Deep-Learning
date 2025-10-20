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
    pl.seed_everything(42, workers=True)

    # Data
    dm = OxfordPetDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        classes=args.classes,
        use_augmentation=args.use_augmentation
    )

    # Model
    num_classes = 3 if args.classes == 'trimap' else 2
    model = UNetPlusPlusModule(
        classes=num_classes,
        lr=args.lr,
        weight_decay=args.weight_decay,
        encoder_name=args.encoder,
        use_dice_loss=args.use_dice_loss
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
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1
    precision = args.precision if accelerator == 'gpu' else 32

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

    # Data
    dm = OxfordPetDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        classes=args.classes,
        use_augmentation=False
    )
    dm.setup()

    # Model
    model = UNetPlusPlusModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_loader = dm.test_dataloader()
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

    vis_images, vis_gt_masks, vis_pred_masks = [], [], []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)

            # Update metrics
            iou_metric.update(preds, masks)
            dice_metric.update(preds, masks)

            # Save visualization samples
            if len(vis_images) < 12:
                for i in range(imgs.shape[0]):
                    if len(vis_images) >= 12:
                        break
                    vis_images.append(imgs[i])
                    vis_gt_masks.append(masks[i])
                    vis_pred_masks.append(preds[i])

    iou_per_class = iou_metric.compute().cpu().numpy()
    dice_per_class = dice_metric.compute().cpu().numpy()
    mean_iou, mean_dice = iou_per_class.mean(), dice_per_class.mean()

    class_names = ['Pet', 'Background', 'Border'] if args.classes == 'trimap' else ['Background', 'Pet']

    print("\n" + "="*60)
    print("Test Set Results:")
    print("="*60)
    print(f"Mean IoU:  {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print("Per-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {iou_per_class[i]:.4f}")
    print("Per-Class Dice:")
    for i, name in enumerate(class_names):
        print(f"  {name:12s}: {dice_per_class[i]:.4f}")
    print("="*60 + "\n")

    samples_dir = args.output_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    create_comparison_grid(
        images=vis_images,
        gt_masks=vis_gt_masks,
        pred_masks=vis_pred_masks,
        mode=args.classes,
        denorm=True,
        save_path=samples_dir / 'test_comparison_grid.png'
    )

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
    parser.add_argument('--data_root', type=str, default='~/data/oxford-iiit-pet')
    parser.add_argument('--classes', type=str, default='trimap', choices=['trimap', 'binary'])
    parser.add_argument('--img_size', type=int, default=512)

    # Model
    parser.add_argument('--encoder', type=str, default='resnet34',
                        choices=['resnet18','resnet34','resnet50','efficientnet-b0','efficientnet-b3','mobilenet_v2'])

    # Training
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--precision', type=str, default='16-mixed', choices=['32','16-mixed','bf16-mixed'])

    # Logging
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--exp_name', type=str, default='baseline')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='oxpet-segmentation')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train','eval','both'])
    parser.add_argument('--checkpoint', type=str, default=None)

    # New options
    parser.add_argument('--use_dice_loss', action='store_true')
    parser.add_argument('--use_augmentation', action='store_true')

    args = parser.parse_args()

    # Paths
    args.data_root = Path(args.data_root).expanduser()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run
    if args.mode == 'train':
        best_ckpt = train(args)
        print(f"\nTo evaluate: python src/train.py --mode eval --checkpoint {best_ckpt}")
    elif args.mode == 'eval':
        if args.checkpoint is None:
            print("Error: --checkpoint required")
            return
        evaluate(args.checkpoint, args)
    elif args.mode == 'both':
        best_ckpt = train(args)
        evaluate(best_ckpt, args)


if __name__ == '__main__':
    main()
