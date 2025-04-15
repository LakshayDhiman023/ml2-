import torch
from dataset import get_davis_dataloaders
from model import VideoSegmentationModel
from utils import visualize_predictions
import matplotlib.pyplot as plt
import numpy as np

def test_dataset():
    """Test the DAVIS dataset and visualize a few samples"""
    print("Testing dataset loading...")
    train_loader, val_loader = get_davis_dataloaders(
        root_dir='DAVIS-2017-trainval-480p', 
        batch_size=2
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)} samples")
    print(f"Validation dataset size: {len(val_loader.dataset)} samples")
    
    # Get a sample batch
    sample_batch = next(iter(val_loader))
    
    # Print shapes
    print("\nSample batch shapes:")
    print(f"Image batch shape: {sample_batch['image'].shape}")
    print(f"Mask batch shape: {sample_batch['mask'].shape}")
    print(f"Number of boxes: {[boxes.shape for boxes in sample_batch['boxes']]}")
    print(f"Paths: {sample_batch['path']}")
    
    # Visualize a sample
    plt.figure(figsize=(15, 10))
    
    # Get the first image in batch
    image = sample_batch['image'][0].permute(1, 2, 0).numpy()
    mask = sample_batch['mask'][0][0].numpy()
    boxes = sample_batch['boxes'][0].numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Plot image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')
    
    # Plot mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    # Plot image with bounding boxes
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                         fill=False, edgecolor='red', linewidth=2))
    plt.title('Bounding Boxes')
    plt.axis('off')
    
    plt.savefig('dataset_sample.png')
    plt.close()
    print("Sample visualization saved to 'dataset_sample.png'")
    
    return train_loader, val_loader

def test_model(val_loader):
    """Test the model forward pass"""
    print("\nTesting model forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = VideoSegmentationModel(num_classes=1, pretrained_backbone=True)
    model = model.to(device)
    model.eval()
    
    # Get a sample batch
    sample_batch = next(iter(val_loader))
    images = sample_batch['image'].to(device)
    
    print(f"Input shape: {images.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images)
    
    # Print output shapes
    print("\nOutput shapes:")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Number of proposals: {[p.shape for p in outputs['proposals']]}")
    
    if outputs['cls_scores'] is not None:
        print(f"Classification scores shape: {outputs['cls_scores'].shape}")
    else:
        print("No classification scores (likely no valid proposals found)")
        
    if outputs['bbox_preds'] is not None:
        print(f"Bounding box predictions shape: {outputs['bbox_preds'].shape}")
    else:
        print("No bounding box predictions (likely no valid proposals found)")
    
    print("\nModel test completed successfully!")
    
    return model

if __name__ == "__main__":
    train_loader, val_loader = test_dataset()
    model = test_model(val_loader) 