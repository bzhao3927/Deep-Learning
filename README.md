# UNet++ Semantic Segmentation on Oxford-IIIT Pet Dataset

Deep learning project implementing UNet++ with hybrid loss function and data augmentation for semantic segmentation.

**Authors:** Cade Boiney, Ken Lam, Ognian Trajanov, Benjamin Zhao  
**Institution:** Hamilton College  
**Course:** Deep Learning, Fall 2025

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/bzhao3927/Deep-Learning.git
cd Deep-Learning/Assignment2
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download dataset
python oxpet_download_and_viz_fixed.py --root ~/data --split trainval --classes trimap

# 3. Train model (~10 hours)
python src/train.py --mode train --data_root ~/data/oxford-iiit-pet \
  --exp_name depth_6_all_changes --base_channels 32 --depth 6 \
  --deep_supervision --use_attention --use_augmentation --use_hybrid_loss \
  --batch_size 4 --accumulate_grad_batches 4 --epochs 150 --lr 1e-3 \
  --weight_decay 1e-4 --precision 16-mixed --use_wandb

# 4. Evaluate
python src/train.py --mode eval --data_root ~/data/oxford-iiit-pet \
  --checkpoint outputs/checkpoints/best_model.ckpt \
  --deep_supervision --use_attention --batch_size 4
```

---

## Overview

UNet++ implementation with two key improvements:

1. **Enhanced Data Augmentation** - Random flips, rotations, color jitter, blur, and crops
2. **Hybrid Loss Function** - Combines Focal Loss (class imbalance) + Dice Loss (segmentation quality)

**Final Performance:**
- Test mIoU: **80.94%**
- Validation mIoU: **82.1%**
- Training time: **~10 hours** (RTX 3090, early stopped at epoch 131)

---

## Installation

```bash
# Clone repository
git clone https://github.com/bzhao3927/Deep-Learning.git
cd Deep-Learning/Assignment2

# Create environment and install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Optional: Setup W&B
pip install wandb
wandb login
```

**Requirements:** Python 3.8+, CUDA 11.8+, 16GB RAM, 8GB+ GPU

---

## Dataset

Download Oxford-IIIT Pet Dataset:

```bash
mkdir -p ~/data
python oxpet_download_and_viz_fixed.py --root ~/data --split trainval --classes trimap
```

**Dataset structure:**
```
~/data/oxford-iiit-pet/
├── images/              # RGB images
└── annotations/
    ├── trimaps/         # Segmentation masks
    ├── trainval.txt
    └── test.txt
```

**Splits:** 2,944 train / 736 val / 3,669 test samples

---

## Training

Reproduce our results:

```bash
python src/train.py --mode train \
  --data_root ~/data/oxford-iiit-pet \
  --exp_name depth_6_all_changes \
  --base_channels 32 --depth 6 \
  --deep_supervision --use_attention \
  --use_augmentation --use_hybrid_loss \
  --batch_size 4 --accumulate_grad_batches 4 \
  --epochs 150 --lr 1e-3 --weight_decay 1e-4 \
  --precision 16-mixed --use_wandb
```

**Training details:**
- Early stopping at epoch 131 (patience=15)
- ~10 hours on NVIDIA RTX 3090
- Mixed precision (16-bit) training
- Checkpoints saved to `outputs/checkpoints/`

---

## Evaluation

```bash
python src/train.py --mode eval \
  --data_root ~/data/oxford-iiit-pet \
  --checkpoint outputs/checkpoints/unetpp-epoch=131-val_mIoU=0.8210.ckpt \
  --deep_supervision --use_attention --batch_size 4
```

Find your best checkpoint:
```bash
ls -lh outputs/checkpoints/
```

---

## Results

### Test Set Performance

| Class | IoU (%) | Dice (%) |
|-------|---------|----------|
| Pet | 88.06 | 93.65 |
| Background | 93.82 | 96.81 |
| Border | 60.95 | 75.74 |
| **Mean** | **80.94** | **88.73** |

### Ablation Study

| Configuration | Val mIoU (%) | Test mIoU (%) |
|---------------|--------------|---------------|
| Baseline (deep supervision + data aug) | 78.3 | 80.00 |
| + Hybrid Loss + CBAM | 82.1 | **80.94** |

**Validation improvement: +3.8% | Test improvement: +0.94%**

---

## Key Improvements

### 1. Enhanced Data Augmentation
Random flips, ±15° rotations, color jitter, blur, and crops regularize training and expose the model to diverse geometric, noise, and color variations.

### 2. Hybrid Loss (Focal + Dice)
Combines Focal Loss for class imbalance (border class is only ~5% of pixels) with Dice Loss for direct IoU optimization:

$L_{hybrid} = 0.5 \times L_{Focal} + 0.5 \times L_{Dice}$

**Impact:** +3.8% validation mIoU, +0.94% test mIoU over baseline

---

## Architecture

**UNet++** with nested skip connections and deep supervision:
- 6 encoder-decoder levels
- 32 base channels (doubled per level)
- CBAM attention modules in decoder
- AdamW optimizer with cosine annealing

---

## Project Structure

```
Assignment2/
├── README.md
├── requirements.txt
├── oxpet_download_and_viz_fixed.py
├── plot.py
├── src/
│   ├── datamodule_oxpet.py
│   ├── model_unetpp.py
│   ├── train.py
│   └── viz.py
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── samples/
```

---

## Monitoring

**View logs:**
```bash
tail -f training.log
```

**W&B Dashboard:**  
[https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation](https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation)

**Generate training curves:**
```bash
python plot.py  # Creates outputs/miou_curves.png and outputs/loss_curves.png
```

---

## Troubleshooting

**CUDA Out of Memory:**
```bash
--batch_size 2 --accumulate_grad_batches 8
```

**Poor performance:** Ensure all improvements are enabled:
```bash
--deep_supervision --use_attention --use_augmentation --use_hybrid_loss
```

---

## Citation

```bibtex
@misc{zhao2025unetpp,
  author = {Zhao, Benjamin and Boiney, Cade and Lam, Ken and Trajanov, Ognian},
  title = {UNet++ with Hybrid Loss and Data Augmentation for Pet Segmentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/bzhao3927/Deep-Learning/tree/main/Assignment2}
}
```

---

## References

- [UNet++](https://arxiv.org/abs/1807.10165) - Zhou et al., 2018
- [CBAM](https://arxiv.org/abs/1807.06521) - Woo et al., 2018
- [Focal Loss](https://arxiv.org/abs/1708.02002) - Lin et al., 2017
- [Dice Loss](https://arxiv.org/abs/1606.04797) - Milletari et al., 2016
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) - Parkhi et al., 2012

---

## Contact

**Benjamin Zhao** - bzhao@hamilton.edu  
Hamilton College, Deep Learning Course, Fall 2025

**W&B Dashboard:** [View experiments](https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation)