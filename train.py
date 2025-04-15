import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_davis_dataloaders
from model import VideoSegmentationModel
from utils import visualize_predictions, evaluate_model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4, checkpoint_dir='checkpoints'):
    """
    Train the video segmentation model
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function and optimizer
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training progress bar
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            boxes = [b.to(device) for b in batch['boxes']]
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            # Classification loss
            cls_scores = outputs['cls_scores']
            
            # Assuming ground truth class is 1 (object) for all proposals
            # In a real scenario, would need to match proposals with ground truth
            if cls_scores is not None:
                gt_classes = torch.ones(cls_scores.size(0), dtype=torch.long, device=device)
                cls_loss = cls_criterion(cls_scores, gt_classes)
            else:
                cls_loss = torch.tensor(0.0, device=device)
            
            # Regression loss for bounding box refinement
            bbox_preds = outputs['bbox_preds']
            
            # In a real implementation, we would need to match predictions with ground truth boxes
            # and calculate IoU-based losses. For simplicity, we'll use a dummy regression target.
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
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = validate(model, val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            
            print(f'Saved checkpoint at epoch {epoch+1} with validation loss {val_loss:.4f}')
        
        # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
        }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'loss_plot.png'))
    
    return train_losses, val_losses

def validate(model, val_loader):
    """
    Validate the model on validation set
    """
    model.eval()
    val_loss = 0.0
    
    # Loss functions
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            boxes = [b.to(device) for b in batch['boxes']]
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss (same as in training)
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
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def main():
    # Load DAVIS dataset
    batch_size = 8
    train_loader, val_loader = get_davis_dataloaders(
        root_dir='DAVIS-2017-trainval-480p',
        batch_size=batch_size
    )
    
    # Initialize model
    model = VideoSegmentationModel(num_classes=1, pretrained_backbone=True)
    model = model.to(device)
    
    # Train model
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=1,
        learning_rate=1e-4,
        checkpoint_dir='checkpoints'
    )
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, iou_threshold=0.5)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Visualize some predictions
    visualize_predictions(model, val_loader, num_samples=5, output_dir='predictions')

if __name__ == '__main__':
    main() 