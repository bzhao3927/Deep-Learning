import os
import random
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
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

        split_file = self.root / 'annotations' / f'{split}.txt'
        with open(split_file) as f:
            self.filenames = [line.strip().split()[0] for line in f if line.strip()]

        self.img_dir = self.root / 'images'
        self.mask_dir = self.root / 'annotations' / 'trimaps'

        self.num_classes = 3 if classes == 'trimap' else 2

    def __len__(self):
        return len(self.filenames)

    def mask_to_classes(self, mask: Image.Image):
        """Convert trimap values (1,2,3) to class indices (0,1,2)"""
        m = np.array(mask, dtype=np.int64)
        if self.classes == 'trimap':
            m = m - 1  # Ensure 0,1,2
        else:
            pet = (m == 1) | (m == 3)
            m = pet.astype(np.int64)
        # Safety clamp
        m = np.clip(m, 0, self.num_classes - 1)
        return torch.tensor(m, dtype=torch.long)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        img_path = self.img_dir / f"{name}.jpg"
        mask_path = self.mask_dir / f"{name}.png"

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize
        img = TF.resize(img, [self.img_size, self.img_size], interpolation=Image.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=Image.NEAREST)

        # Augmentation (only training)
        if self.augment:
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        mask = self.mask_to_classes(mask)

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
        # Transforms
        train_transform = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        full_ds = OxfordPetDataset(
            self.root, split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=train_transform,
            augment=self.use_augmentation
        )

        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, val_ds = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        # For validation, create a clean dataset (no augmentation, proper transform)
        val_full = OxfordPetDataset(
            self.root, split='trainval',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform,
            augment=False
        )
        self.val_ds = Subset(val_full, val_ds.indices)
        self.train_ds = train_ds

        # Test dataset
        self.test_ds = OxfordPetDataset(
            self.root, split='test',
            img_size=self.img_size,
            classes=self.classes,
            transform=val_transform,
            augment=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
