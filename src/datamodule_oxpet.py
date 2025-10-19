import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import pytorch_lightning as pl

class OxfordPetDataset(Dataset):
    def __init__(self, root, split='trainval', img_size=512, classes='trimap', transform=None, augment=False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.classes = classes
        self.transform = transform
        self.augment = augment

        # Load filenames from split file
        split_file = self.root / 'annotations' / f'{split}.txt'
        with open(split_file) as f:
            lines = f.readlines()
            self.filenames = []
            for line in lines:
                line = line.strip()
                if line:
                    filename = line.split()[0]
                    self.filenames.append(filename)

        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'annotations' / 'trimaps'

    def __len__(self):
        return len(self.filenames)

    def mask_to_classes(self, mask: Image.Image):
        """Convert trimap values (1,2,3) to class indices (0,1,2)"""
        m = np.array(mask, dtype=np.int64)
        if self.classes == 'trimap':
            m = m - 1  # 0=pet, 1=background, 2=border
        else:
            pet = (m == 1) | (m == 3)
            m = pet.astype(np.int64)
        return torch.tensor(m, dtype=torch.long)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize
        img = TF.resize(img, [self.img_size, self.img_size], 
                       interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], 
                        interpolation=Image.NEAREST)

        # Apply augmentation if enabled (for training only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            
            # Random rotation (-15 to +15 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Convert mask to class indices
        mask = self.mask_to_classes(mask)

        # Apply transforms to image (ToTensor + ColorJitter + Normalize)
        if self.transform:
            img = self.transform(img)

        return img, mask

class OxfordPetDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=8, img_size=512, num_workers=4, classes='trimap', use_augmentation=False):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.classes = classes
        self.use_augmentation = use_augmentation

    def setup(self, stage=None):
        # Training transforms WITH augmentation
        train_transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Val/Test transforms WITHOUT augmentation
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Load full trainval dataset
        full_ds = OxfordPetDataset(
            self.root, 
            split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=train_transform,
            augment=self.use_augmentation  # Only augment if flag is True
        )
        
        # Split train/val 80/20 with fixed seed
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(
            full_ds, 
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Wrap train dataset to use augmentation
        self.train_ds = train_ds
        
        # Create separate validation dataset WITHOUT augmentation
        val_ds_no_aug = OxfordPetDataset(
            self.root, 
            split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform,
            augment=False  # No augmentation for validation
        )
        # Use same indices as validation split
        self.val_ds = torch.utils.data.Subset(val_ds_no_aug, val_ds.indices)

        # Test dataset (no augmentation)
        self.test_ds = OxfordPetDataset(
            self.root, 
            split='test',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True
        )