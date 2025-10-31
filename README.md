# UNet++ Semantic Segmentation on Oxford-IIIT Pet Dataset

Deep learning project implementing UNet++ with hybrid loss function and CBAM attention for semantic segmentation.

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
  --exp_name depth_6_all_changes \
  --base_channels 32 --depth 6 --deep_supervision --use_attention \
  --use_augmentation --use_hybrid_loss --batch_size 4 \
  --accumulate_grad_batches 4 --epochs 150 --lr 1e-3 --weight_decay 1e-4 \
  --precision 16-mixed --use_wandb --wandb_project unetpp-oxpet-segmentation

# 4. Evaluate
python src/train.py --mode eval \
  --data_root ~/data/oxford-iiit-pet \
  --checkpoint outputs/checkpoints/best_model.ckpt \
  --deep_supervision --use_attention --batch_size 4
```

**See full documentation below for details.**

---

## Overview

This project implements **UNet++** enhanced with two key improvements:

1. **Hybrid Loss Function** - Combines Focal Loss (class imbalance) + Dice Loss (segmentation quality)
2. **CBAM Attention** - Channel and Spatial attention modules for improved boundary localization

**Final Performance:**
- Test mIoU: **80.94%**
- Validation mIoU: **82.1%** (epoch 131)
- Training time: **~10 hours** (single GPU with early stopping at epoch 131)

---

## Features

**Architecture:** UNet++ with nested skip connections and deep supervision  
**Depth:** 6 encoder-decoder levels  
**Attention:** CBAM (Channel + Spatial) modules in decoder  
**Loss:** Hybrid: 0.5×Focal + 0.5×Dice  
**Optimizer:** AdamW with cosine annealing  
**Framework:** PyTorch Lightning  
**Logging:** Weights & Biases

---

## Requirements

**System:**
- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- 8GB+ GPU memory recommended

**Python Dependencies:**
```
torch>=2.1.0
torchvision>=0.16.0
pytorch-lightning>=2.2.0
numpy>=1.25.0
Pillow>=10.0.0
matplotlib>=3.8.0
segmentation-models-pytorch>=0.3.1
torchmetrics>=1.0.0
```

**Optional:**
```
wandb>=0.16.0  # For experiment tracking
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/bzhao3927/Deep-Learning.git
cd Deep-Learning/Assignment2
```

### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# On Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Weights & Biases (optional)
```bash
pip install wandb
wandb login
```

---

## Dataset Preparation

### Download Oxford-IIIT Pet Dataset
```bash
# Create data directory
mkdir -p ~/data

# Run the provided download script
python oxpet_download_and_viz_fixed.py \
  --root ~/data \
  --split trainval \
  --classes trimap \
  --n 12 \
  --resize 512
```

**Expected directory structure:**
```
~/data/oxford-iiit-pet/
├── images/                # RGB images (.jpg)
└── annotations/
    ├── trimaps/           # Segmentation masks (.png)
    ├── trainval.txt       # Training/validation split
    └── test.txt           # Test split
```

---

## Usage

### Training

To reproduce our results, run:
```bash
python src/train.py \
  --mode train \
  --data_root ~/data/oxford-iiit-pet \
  --exp_name depth_6_all_changes \
  --base_channels 32 \
  --depth 6 \
  --deep_supervision \
  --use_attention \
  --use_augmentation \
  --use_hybrid_loss \
  --batch_size 4 \
  --accumulate_grad_batches 4 \
  --epochs 150 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --precision 16-mixed \
  --use_wandb \
  --wandb_project unetpp-oxpet-segmentation
```

**Training details:**
- Runs for up to 150 epochs with early stopping (patience=15)
- Actual training time: **~10 hours** on NVIDIA GeForce RTX 5070 Ti (stopped at epoch 131)
- Validation mIoU peaked around 82.1% and plateaued, triggering early stopping
- Saved ~5 hours by stopping at epoch 131 instead of running full 150 epochs
- Checkpoints saved to `outputs/checkpoints/`
- Logs saved to `outputs/logs/`

### Evaluation

After training completes, evaluate on the test set:
```bash
python src/train.py \
  --mode eval \
  --data_root ~/data/oxford-iiit-pet \
  --checkpoint outputs/checkpoints/unetpp-epoch=131-val_mIoU=0.8210.ckpt \
  --deep_supervision \
  --use_attention \
  --batch_size 4 \
  --use_wandb \
  --wandb_project unetpp-oxpet-segmentation
```

**Note:** Replace the checkpoint filename with your actual best checkpoint. To find it:
```bash
ls -lh outputs/checkpoints/
# Look for the checkpoint with highest val_mIoU
```

---

## Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_root` | Path to dataset | `~/data/oxford-iiit-pet` |
| `--exp_name` | Experiment name for logging | `unetpp_experiment` |
| `--base_channels` | Base number of channels | `32` |
| `--depth` | Encoder/decoder depth | `5` |
| `--deep_supervision` | Enable deep supervision | `False` |
| `--use_attention` | Enable CBAM attention | `False` |
| `--use_augmentation` | Enable heavy augmentation | `False` |
| `--use_hybrid_loss` | Use Focal+Dice loss | `False` |
| `--batch_size` | Batch size | `8` |
| `--accumulate_grad_batches` | Gradient accumulation steps | `1` |
| `--epochs` | Training epochs | `50` |
| `--lr` | Learning rate | `1e-3` |
| `--precision` | Mixed precision mode | `16-mixed` |

---

## Results

### Quantitative Results (Test Set)

| Class | IoU (%) | Dice (%) |
|-------|---------|----------|
| Pet | 88.06 | 93.65 |
| Background | 93.82 | 96.81 |
| Border | 60.95 | 75.74 |
| **Mean** | **80.94** | **88.73** |

### Ablation Study

| Configuration | Val mIoU (%) | Test mIoU (%) | Improvement |
|---------------|--------------|---------------|-------------|
| Baseline (deep supervision + data aug) | 78.3 | 80.00 | - |
| + Hybrid Loss | TBD | TBD | TBD |
| + Hybrid Loss + CBAM | 82.1 | **80.94** | +0.94% |

**Validation improvement: +3.8% over baseline**

### Visualizations

After evaluation, sample predictions are saved to `outputs/samples/`:
- `test_comparison_grid.png` - Grid of 12 test samples
- Individual sample predictions

Each visualization shows: input image | ground truth mask | prediction | overlay

---

## Project Structure
```
Assignment2/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── oxpet_download_and_viz_fixed.py    # Dataset download script
├── plot.py                            # Script to generate training curves
│
├── src/
│   ├── datamodule_oxpet.py            # PyTorch Lightning DataModule
│   ├── model_unetpp.py                # UNet++ LightningModule
│   ├── train.py                       # Training/evaluation script
│   └── viz.py                         # Visualization utilities
│
└── outputs/
    ├── checkpoints/                   # Saved model checkpoints
    ├── logs/                          # Training logs
    └── samples/                       # Prediction visualizations
```

---

## Monitoring Training

### View logs in real-time

If running in background:
```bash
# Start training in background
nohup python src/train.py --mode train [args] > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### View in Weights & Biases

Visit: [https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation](https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation)

### Check saved checkpoints
```bash
ls -lh outputs/checkpoints/
```

---

## Generating Training Curves

To regenerate the training curve plots from your W&B data:

1. Export your training data from W&B as CSV files:
   - `miou.csv` - Contains epoch, train_mIoU, val_mIoU
   - `loss.csv` - Contains step, train_loss, val_loss

2. Place the CSV files in the `Assignment2/` directory

3. Run the plotting script:
```bash
python plot.py
```

This will create high-resolution plots in `outputs/`:
- `outputs/miou_curves.png` - mIoU over epochs
- `outputs/loss_curves.png` - Loss over training steps

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size and increase gradient accumulation:
```bash
--batch_size 2 --accumulate_grad_batches 8
```

### Training too slow

Mixed precision is already enabled by default (`--precision 16-mixed`). For faster iteration during development, reduce epochs:
```bash
--epochs 50
```

### Poor performance

Ensure all improvements are enabled:
```bash
--deep_supervision --use_attention --use_augmentation --use_hybrid_loss
```

### FileNotFoundError for dataset

Verify dataset path and structure:
```bash
ls ~/data/oxford-iiit-pet/images/
ls ~/data/oxford-iiit-pet/annotations/trimaps/
```

---

## Key Improvements

### Improvement 1: Hybrid Loss Function (Focal + Dice)

Combines Focal Loss for handling class imbalance with Dice Loss for direct IoU optimization. The dataset exhibits severe class imbalance with border pixels representing only approximately 5% of the total, making this hybrid approach particularly effective.

**Mathematical formulation:**
```
L_hybrid = 0.5 × L_Focal + 0.5 × L_Dice
```

**Impact:** Substantial improvement on validation (+3.8% mIoU), with modest gains on test (+0.94% mIoU)

### Improvement 2: CBAM Attention

Integrates Convolutional Block Attention Module (CBAM) which sequentially applies channel attention (CAM) to weight feature channels based on global context, followed by spatial attention (SAM) to focus on informative spatial locations for accurate boundary delineation.

**Impact:** Improved boundary localization and feature discrimination, particularly visible in challenging cases with occlusions or cluttered backgrounds

---

## GPU and Training Requirements

**Tested Hardware:**
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- RAM: 32GB system memory
- Storage: ~10GB for dataset + outputs

**Training Time:**
- With early stopping (actual): **~10 hours** (stopped at epoch 131)
- Full 150 epochs (if no early stopping): 15-20 hours
- Early stopping patience: 15 epochs
- Validation mIoU at epoch 131: 82.1%

**GPU Utilization:**
- Memory usage: ~8-10GB VRAM
- Utilization: 60-80% during training

---

## Citation

If you use this code, please cite:
```bibtex
@misc{zhao2025unetpp,
  author = {Zhao, Benjamin and Boiney, Cade and Lam, Ken and Trajanov, Ognian},
  title = {UNet++ with Hybrid Loss and CBAM Attention for Pet Segmentation},
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

**Benjamin Zhao** - [bzhao@hamilton.edu](mailto:bzhao@hamilton.edu)  
**Cade Boiney** - Hamilton College  
**Ken Lam** - Hamilton College  
**Ognian Trajanov** - Hamilton College

Hamilton College, Deep Learning Course, Fall 2025

---

## W&B Dashboard

View complete training results and experiments:  
[https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation](https://wandb.ai/bzhao-hamilton-college/unetpp-oxpet-segmentation)

---

## License

This project is for educational purposes as part of Hamilton College coursework.
