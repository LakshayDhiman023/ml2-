import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DAVISDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        DAVIS dataset for video object segmentation
        Args:
            root_dir (string): Directory with DAVIS dataset
            split (string): 'train' or 'val' split
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Default paths for the DAVIS dataset structure
        self.img_dir = os.path.join(root_dir, 'DAVIS', 'JPEGImages', '480p')
        self.mask_dir = os.path.join(root_dir, 'DAVIS', 'Annotations', '480p')
        
        # Get split information
        splits_dir = os.path.join(root_dir, 'DAVIS', 'ImageSets', '2017')
        
        if split == 'train':
            split_file = os.path.join(splits_dir, 'train.txt')
        elif split == 'val':
            split_file = os.path.join(splits_dir, 'val.txt')
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'val'.")
        
        # Read sequences for this split
        try:
            with open(split_file, 'r') as f:
                self.sequences = [line.strip() for line in f]
        except FileNotFoundError:
            # Fallback to just listing directories if split file not found
            self.sequences = sorted(os.listdir(self.img_dir))
        
        # Build list of all frame paths and corresponding masks
        self.frames = []
        self.masks = []
        
        for seq in self.sequences:
            seq_frames = sorted(os.listdir(os.path.join(self.img_dir, seq)))
            for frame in seq_frames:
                if frame.endswith('.jpg'):
                    frame_path = os.path.join(self.img_dir, seq, frame)
                    mask_name = frame.replace('.jpg', '.png')
                    mask_path = os.path.join(self.mask_dir, seq, mask_name)
                    
                    if os.path.exists(mask_path):
                        self.frames.append(frame_path)
                        self.masks.append(mask_path)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        img_path = self.frames[idx]
        mask_path = self.masks[idx]
        
        # Read image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Get bounding box from mask (for region proposals)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize empty tensor for bounding boxes
        boxes = torch.zeros((len(contours), 4), dtype=torch.float32)
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            boxes[i, 0] = x
            boxes[i, 1] = y
            boxes[i, 2] = x + w
            boxes[i, 3] = y + h
        
        # If no contours found, create a dummy box
        if len(contours) == 0:
            boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        
        sample = {
            'image': image,
            'mask': mask,
            'boxes': boxes,
            'path': img_path
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class DAVISTransform:
    def __init__(self, size=(224, 224)):
        self.size = size
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, sample):
        image, mask, boxes, path = sample['image'], sample['mask'], sample['boxes'], sample['path']
        
        # Resize image and mask
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        # Scale bounding boxes to match new image size
        orig_h, orig_w = sample['image'].shape[:2]
        new_h, new_w = self.size
        
        boxes[:, 0] = boxes[:, 0] * (new_w / orig_w)
        boxes[:, 1] = boxes[:, 1] * (new_h / orig_h)
        boxes[:, 2] = boxes[:, 2] * (new_w / orig_w)
        boxes[:, 3] = boxes[:, 3] * (new_h / orig_h)
        
        # Convert image to tensor and normalize
        image_tensor = self.img_transform(image)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'boxes': boxes,
            'path': path
        }

def custom_collate_fn(batch):
    """
    Custom collate function to handle samples with different numbers of boxes
    """
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    # Don't stack boxes, just return them as a list
    boxes = [item['boxes'] for item in batch]
    paths = [item['path'] for item in batch]
    
    return {
        'image': images,
        'mask': masks,
        'boxes': boxes,
        'path': paths
    }

def get_davis_dataloaders(root_dir='DAVIS-2017-trainval-480p', batch_size=8):
    """
    Create train and validation dataloaders for DAVIS dataset
    """
    transform = DAVISTransform(size=(224, 224))
    
    train_dataset = DAVISDataset(root_dir=root_dir, split='train', transform=transform)
    val_dataset = DAVISDataset(root_dir=root_dir, split='val', transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader 