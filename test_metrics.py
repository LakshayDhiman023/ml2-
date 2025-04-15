import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import random
from tqdm import tqdm

from dataset import DAVISDataset, DAVISTransform, custom_collate_fn
from model import VideoSegmentationModel
from utils import (
    evaluate_model, 
    evaluate_segmentation_metrics,
    calculate_precision_recall_f_measure,
    calculate_s_measure,
    calculate_mae,
    masks_from_boxes
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_small_subset(dataset, num_samples=10):
    """Create a small subset of the dataset for testing"""
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    return Subset(dataset, indices)

def test_metrics():
    """Test the segmentation metrics on a small subset of the dataset"""
    # Parameters
    root_dir = 'DAVIS-2017-trainval-480p'
    batch_size = 2
    num_samples = 10  # Number of samples to use for testing
    
    # Create transform
    transform = DAVISTransform(size=(224, 224))
    
    # Load dataset
    print("Loading dataset...")
    try:
        val_dataset = DAVISDataset(root_dir=root_dir, split='val', transform=transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create small subset
    val_subset = create_small_subset(val_dataset, num_samples)
    print(f"Created validation subset with {len(val_subset)} samples")
    
    # Create data loader
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = VideoSegmentationModel(num_classes=1, pretrained_backbone=True)
    model = model.to(device)
    model.eval()
    
    # Test individual metrics on a single example
    print("\nTesting individual metrics on a single example...")
    sample_batch = next(iter(val_loader))
    
    gt_mask = sample_batch['mask'][0][0].numpy()
    
    # Create a dummy prediction mask (random for testing)
    h, w = gt_mask.shape
    rand_pred = np.random.rand(h, w) > 0.5
    
    # Convert the sample's bounding boxes to a mask
    boxes = sample_batch['boxes'][0].numpy()
    box_mask = masks_from_boxes(boxes, (h, w))
    
    # Add some noise to simulate imperfect prediction
    noise = np.random.rand(h, w) > 0.9  # 10% noise
    noisy_box_mask = box_mask.copy()
    noisy_box_mask[noise] = 1 - noisy_box_mask[noise]
    
    # Calculate metrics
    p, r, f = calculate_precision_recall_f_measure(box_mask, gt_mask)
    print(f"Ground Truth vs Box Mask:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F-measure: {f:.4f}")
    
    p, r, f = calculate_precision_recall_f_measure(noisy_box_mask, gt_mask)
    print(f"Ground Truth vs Noisy Box Mask:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F-measure: {f:.4f}")
    
    p, r, f = calculate_precision_recall_f_measure(rand_pred, gt_mask)
    print(f"Ground Truth vs Random Prediction:")
    print(f"  Precision: {p:.4f}")
    print(f"  Recall: {r:.4f}")
    print(f"  F-measure: {f:.4f}")
    
    # Calculate S-measure
    s = calculate_s_measure(box_mask, gt_mask)
    print(f"S-measure (GT vs Box): {s:.4f}")
    
    s = calculate_s_measure(noisy_box_mask, gt_mask)
    print(f"S-measure (GT vs Noisy Box): {s:.4f}")
    
    s = calculate_s_measure(rand_pred, gt_mask)
    print(f"S-measure (GT vs Random): {s:.4f}")
    
    # Calculate MAE
    mae = calculate_mae(box_mask, gt_mask)
    print(f"MAE (GT vs Box): {mae:.4f}")
    
    mae = calculate_mae(noisy_box_mask, gt_mask)
    print(f"MAE (GT vs Noisy Box): {mae:.4f}")
    
    mae = calculate_mae(rand_pred, gt_mask)
    print(f"MAE (GT vs Random): {mae:.4f}")
    
    # Evaluate on whole subset
    print("\nEvaluating metrics on the entire subset...")
    
    # Run full evaluation with all metrics
    metrics = evaluate_model(model, val_loader)
    
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\nMetrics evaluation completed!")

if __name__ == "__main__":
    test_metrics() 