# src/viz.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms.functional as TF
from pathlib import Path

# Color palette for trimap classes
TRIMAP_COLORS = {
    0: (220, 20, 60),    # Pet (Crimson Red)
    1: (119, 135, 150),  # Background (Gray)
    2: (34, 139, 34)     # Border (Forest Green)
}

BINARY_COLORS = {
    0: (119, 135, 150),  # Background (Gray)
    1: (220, 20, 60)     # Pet (Crimson Red)
}

CLASS_NAMES = {
    'trimap': ['Pet', 'Background', 'Border'],
    'binary': ['Background', 'Pet']
}


def colorize_mask(mask, mode='trimap'):
    """
    Convert a segmentation mask to RGB visualization.
    
    Args:
        mask: numpy array or torch tensor of shape (H, W) with class indices
        mode: 'trimap' (3 classes) or 'binary' (2 classes)
    
    Returns:
        PIL Image in RGB format
    """
    # Convert to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    mask = mask.astype(np.int64)
    H, W = mask.shape
    
    # Create RGB image
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Apply colors based on mode
    colors = TRIMAP_COLORS if mode == 'trimap' else BINARY_COLORS
    for class_idx, color in colors.items():
        rgb[mask == class_idx] = color
    
    return Image.fromarray(rgb, mode='RGB')


def overlay_mask(image, mask_rgb, alpha=0.45):
    """
    Overlay a colored mask on top of an image.
    
    Args:
        image: PIL Image or torch tensor (C, H, W) in range [0, 1]
        mask_rgb: PIL Image (colored mask)
        alpha: transparency of mask overlay (0=invisible, 1=opaque)
    
    Returns:
        PIL Image with overlaid mask
    """
    # Convert tensor to PIL if needed
    if isinstance(image, torch.Tensor):
        image = TF.to_pil_image(image.cpu())
    
    # Ensure both are PIL Images
    image = image.convert('RGB')
    
    # Resize mask to match image size if needed
    if image.size != mask_rgb.size:
        mask_rgb = mask_rgb.resize(image.size, resample=Image.NEAREST)
    
    # Blend images
    return Image.blend(image, mask_rgb, alpha)


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize an image tensor back to [0, 1] range.
    
    Args:
        tensor: torch tensor (C, H, W) normalized with ImageNet stats
        mean: normalization mean used during training
        std: normalization std used during training
    
    Returns:
        torch tensor (C, H, W) in range [0, 1]
    """
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def create_comparison_grid(images, gt_masks, pred_masks, mode='trimap', 
                           denorm=True, save_path=None):
    """
    Create a grid showing: Image | GT Mask | Pred Mask | Overlay for multiple samples.
    
    Args:
        images: list of torch tensors (C, H, W) or PIL Images
        gt_masks: list of ground truth masks (H, W)
        pred_masks: list of predicted masks (H, W)
        mode: 'trimap' or 'binary'
        denorm: whether to denormalize images (if they're normalized tensors)
        save_path: optional path to save the grid
    
    Returns:
        PIL Image containing the comparison grid
    """
    n_samples = len(images)
    rows = []
    
    for i in range(n_samples):
        img = images[i]
        
        # Denormalize if needed
        if isinstance(img, torch.Tensor) and denorm:
            img = denormalize_image(img)
        
        # Convert to PIL
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img.cpu())
        
        # Colorize masks
        gt_colored = colorize_mask(gt_masks[i], mode=mode)
        pred_colored = colorize_mask(pred_masks[i], mode=mode)
        
        # Create overlays
        gt_overlay = overlay_mask(img.copy(), gt_colored, alpha=0.45)
        pred_overlay = overlay_mask(img.copy(), pred_colored, alpha=0.45)
        
        # Create row: [Image, GT, Pred, GT Overlay, Pred Overlay]
        row_images = [img, gt_colored, pred_colored, gt_overlay, pred_overlay]
        
        # Ensure all same size
        size = img.size
        row_images = [im.resize(size, Image.NEAREST) if im.size != size else im 
                     for im in row_images]
        
        # Concatenate horizontally
        row = Image.new('RGB', (size[0] * 5, size[1]))
        for j, im in enumerate(row_images):
            row.paste(im, (j * size[0], 0))
        
        rows.append(row)
    
    # Concatenate all rows vertically
    if rows:
        width = rows[0].size[0]
        height = sum(row.size[1] for row in rows)
        grid = Image.new('RGB', (width, height))
        
        y_offset = 0
        for row in rows:
            grid.paste(row, (0, y_offset))
            y_offset += row.size[1]
        
        # Save if path provided
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            grid.save(save_path)
            print(f"Saved visualization to {save_path}")
        
        return grid
    
    return None


def save_single_prediction(image, gt_mask, pred_mask, save_path, mode='trimap', denorm=True):
    """
    Save a single prediction with image, GT, prediction, and overlays.
    
    Args:
        image: torch tensor (C, H, W) or PIL Image
        gt_mask: ground truth mask (H, W)
        pred_mask: predicted mask (H, W)
        save_path: where to save the result
        mode: 'trimap' or 'binary'
        denorm: whether to denormalize image
    """
    grid = create_comparison_grid(
        [image], [gt_mask], [pred_mask],
        mode=mode, denorm=denorm, save_path=save_path
    )
    return grid