"""
src/datamodule_oxpet.py
Oxford-IIIT Pet DataModule with Albumentations
Includes IMPROVEMENT 1: Augmentation
"""

import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ============================================================================
# IMPROVEMENT 1: Augmentation Transforms
# ============================================================================

def get_baseline_transforms(img_size=512):
    """No augmentation - pure baseline"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_augmentation_transforms(img_size=512):
    """Augmentation pipeline optimized for semantic segmentation
    
    Key design choices:
    - No VerticalFlip (unnatural for pets)
    - Moderate rotation (±10°) to preserve spatial relationships
    - Gentler color variations to avoid distribution shift
    - Light quality augmentation for robustness
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        
        # Spatial transforms - moderate intensity
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,      # Slight translation
            scale_limit=0.1,         # 90-110% scale
            rotate_limit=10,         # ±10° rotation
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3
        ),
        
        # Color transforms - gentle
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
            p=0.4
        ),
        
        # Quality/robustness - light application
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 20.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.15),
        
        # Always last
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms(img_size=512):
    """No augmentation for validation/test"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ============================================================================
# Dataset
# ============================================================================

class OxfordPetDataset(Dataset):
    """Oxford-IIIT Pet Dataset with Albumentations support"""
    
    def __init__(self, root, split='trainval', img_size=512, classes='trimap', transform=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.classes = classes
        self.transform = transform

        # Load filenames from split file
        split_file = self.root / 'annotations' / f'{split}.txt'
        with open(split_file) as f:
            self.filenames = [line.strip().split()[0] for line in f if line.strip()]

        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'annotations' / 'trimaps'
        self.num_classes = 3 if classes == 'trimap' else 2

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        # Load as numpy arrays (required by Albumentations)
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Convert mask: trimap (1,2,3) → class indices (0,1,2)
        if self.classes == 'trimap':
            mask = mask - 1
        else:
            pet = (mask == 1) | (mask == 3)
            mask = pet.astype(np.int64)
        
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply Albumentations transforms
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
        
        # FIXED: Use as_tensor instead of tensor
        mask = torch.as_tensor(mask, dtype=torch.long)
        
        return img, mask


# ============================================================================
# DataModule
# ============================================================================

class OxfordPetDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Oxford-IIIT Pet"""
    
    def __init__(self, root, batch_size=8, img_size=512, num_workers=4, 
                 classes='trimap', use_augmentation=False):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.classes = classes
        self.use_augmentation = use_augmentation

    def setup(self, stage=None):
        # Choose augmentation strategy
        if self.use_augmentation:
            train_transform = get_augmentation_transforms(self.img_size)
            print("IMPROVEMENT 1: Using augmentation transforms")
        else:
            train_transform = get_baseline_transforms(self.img_size)
            print("Using BASELINE transforms")
        
        val_transform = get_val_transforms(self.img_size)

        # Create full training dataset
        full_ds = OxfordPetDataset(
            self.root, 
            split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=train_transform
        )

        # Split into train/val (80/20)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds_temp = random_split(
            full_ds, 
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Create clean validation dataset (no augmentation)
        val_full = OxfordPetDataset(
            self.root, 
            split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform
        )
        self.val_ds = Subset(val_full, val_ds_temp.indices)
        self.train_ds = train_ds

        # Test dataset
        self.test_ds = OxfordPetDataset(
            self.root, 
            split='test',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform
        )
        
        print(f"Train samples: {len(self.train_ds)}")
        print(f"Val samples: {len(self.val_ds)}")
        print(f"Test samples: {len(self.test_ds)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers, 
            pin_memory=True
        )