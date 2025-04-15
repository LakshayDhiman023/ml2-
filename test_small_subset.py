import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from dataset import DAVISDataset, DAVISTransform, custom_collate_fn
from model import VideoSegmentationModel
from utils import visualize_predictions, evaluate_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def create_small_subset(dataset, num_samples=20):
    """Create a small subset of the dataset for testing"""
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), min(num_samples, total_samples))
    return Subset(dataset, indices)

def test_and_train_small_subset():
    """Test and train the model on a small subset of the dataset"""
    # Parameters
    root_dir = 'DAVIS-2017-trainval-480p'
    batch_size = 4
    num_samples = 20  # Number of samples to use for testing
    num_epochs = 2    # Number of epochs to train
    
    # Create transform
    transform = DAVISTransform(size=(224, 224))
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_dataset = DAVISDataset(root_dir=root_dir, split='train', transform=transform)
        val_dataset = DAVISDataset(root_dir=root_dir, split='val', transform=transform)
    except Exception as e:
        print(f"Error loading full dataset: {e}")
        print("Trying to list image and mask directories...")
        img_dir = os.path.join(root_dir, 'DAVIS', 'JPEGImages', '480p')
        mask_dir = os.path.join(root_dir, 'DAVIS', 'Annotations', '480p')
        
        if os.path.exists(img_dir):
            print(f"Image directory exists, contents: {os.listdir(img_dir)[:5]}")
        else:
            print(f"Image directory {img_dir} does not exist")
        
        if os.path.exists(mask_dir):
            print(f"Mask directory exists, contents: {os.listdir(mask_dir)[:5]}")
        else:
            print(f"Mask directory {mask_dir} does not exist")
        return
    
    # Create small subsets
    train_subset = create_small_subset(train_dataset, num_samples)
    val_subset = create_small_subset(val_dataset, num_samples)
    
    print(f"Created small training subset with {len(train_subset)} samples")
    print(f"Created small validation subset with {len(val_subset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for easier debugging
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for easier debugging
        collate_fn=custom_collate_fn
    )
    
    # Initialize model
    print("Initializing model...")
    model = VideoSegmentationModel(num_classes=1, pretrained_backbone=True)
    model = model.to(device)
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    try:
        sample_batch = next(iter(val_loader))
        images = sample_batch['image'].to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        print("Forward pass successful!")
        print(f"Output features shape: {outputs['features'].shape}")
        print(f"Number of proposals: {[p.shape for p in outputs['proposals']]}")
        
        if outputs['cls_scores'] is not None:
            print(f"Classification scores shape: {outputs['cls_scores'].shape}")
        else:
            print("No classification scores (likely no valid proposals found)")
    except Exception as e:
        print(f"Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # If forward pass is successful, try training for a few epochs
    print("\nTesting training loop for a few epochs...")
    
    # Loss functions and optimizer
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train for a few epochs
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['image'].to(device)
                boxes = [b.to(device) for b in batch['boxes']]
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                cls_scores = outputs['cls_scores']
                
                if cls_scores is not None:
                    gt_classes = torch.ones(cls_scores.size(0), dtype=torch.long, device=device)
                    cls_loss = cls_criterion(cls_scores, gt_classes)
                else:
                    cls_loss = torch.tensor(0.0, device=device)
                
                bbox_preds = outputs['bbox_preds']
                
                if bbox_preds is not None:
                    dummy_reg_targets = torch.zeros_like(bbox_preds)
                    reg_loss = reg_criterion(bbox_preds, dummy_reg_targets)
                else:
                    reg_loss = torch.tensor(0.0, device=device)
                
                # Total loss
                loss = cls_loss + reg_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': epoch_loss / (batch_idx + 1)})
            except Exception as e:
                print(f"Error during training: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"Epoch {epoch+1}/{num_epochs} completed with average loss: {epoch_loss / len(train_loader):.4f}")
    
    # Visualize some predictions
    os.makedirs('test_predictions', exist_ok=True)
    try:
        visualize_predictions(model, val_loader, num_samples=3, output_dir='test_predictions')
        print("Saved prediction visualizations to 'test_predictions' directory")
    except Exception as e:
        print(f"Error visualizing predictions: {e}")
        import traceback
        traceback.print_exc()
    
    print("Small subset testing completed!")

if __name__ == "__main__":
    test_and_train_small_subset() 