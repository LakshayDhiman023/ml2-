import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def visualize_predictions(model, dataloader, num_samples=5, output_dir='predictions'):
    """
    Visualize model predictions on a few sample images
    Args:
        model: trained model
        dataloader: validation dataloader
        num_samples: number of samples to visualize
        output_dir: directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get samples
    samples_visualized = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Generating predictions'):
            images = batch['image'].to(device)
            gt_boxes = batch['boxes']
            paths = batch['path']
            
            # Get predictions
            outputs = model(images)
            pred_boxes = outputs['proposals']
            cls_scores = outputs['cls_scores']
            
            # Visualize each image in the batch
            for i in range(images.shape[0]):
                if samples_visualized >= num_samples:
                    break
                
                # Get the image
                img = images[i].cpu().permute(1, 2, 0).numpy()
                
                # Denormalize the image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                # Get ground truth boxes
                gt_box = gt_boxes[i].cpu().numpy()
                
                # Get predicted boxes
                pred_box = pred_boxes[i].cpu().numpy() if i < len(pred_boxes) else np.array([])
                
                # Create figure
                fig, ax = plt.subplots(1, figsize=(10, 10))
                ax.imshow(img)
                
                # Plot ground truth boxes (green)
                for box in gt_box:
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='g',
                        facecolor='none',
                        label='Ground Truth'
                    )
                    ax.add_patch(rect)
                
                # Plot predicted boxes (red)
                for j, box in enumerate(pred_box):
                    rect = patches.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        linewidth=2,
                        edgecolor='r',
                        facecolor='none',
                        label='Prediction' if j == 0 else None
                    )
                    ax.add_patch(rect)
                
                # Get the image path and extract the filename
                img_path = paths[i]
                img_name = os.path.basename(img_path).split('.')[0]
                
                # Add title
                ax.set_title(f'Image: {img_name}')
                
                # Remove axis
                ax.axis('off')
                
                # Add legend (once)
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'pred_{samples_visualized:03d}_{img_name}.png'))
                plt.close()
                
                samples_visualized += 1
            
            if samples_visualized >= num_samples:
                break
    
    print(f'Saved {samples_visualized} prediction visualizations to {output_dir}')

def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        IoU score
    """
    # Get coordinates of intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection_area = width * height
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_precision_recall_f_measure(pred_mask, gt_mask, beta=1):
    """
    Calculate precision, recall and F-measure between prediction and ground truth masks
    Args:
        pred_mask: predicted binary mask
        gt_mask: ground truth binary mask
        beta: weight of precision in F-measure, default is 1 (F1 score)
    Returns:
        precision, recall, f_measure
    """
    # Ensure masks are binary (0 or 1)
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Calculate true positives, false positives, false negatives
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask) - tp
    fn = np.sum(gt_mask) - tp
    
    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Calculate F-measure
    beta_square = beta * beta
    f_measure = (1 + beta_square) * precision * recall / (beta_square * precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f_measure

def calculate_s_measure(pred_mask, gt_mask, alpha=0.5):
    """
    Calculate S-measure (Structure measure) between prediction and ground truth masks
    Args:
        pred_mask: predicted binary mask
        gt_mask: ground truth binary mask
        alpha: balance factor for region and object similarity
    Returns:
        s_measure
    """
    # Ensure masks are binary (0 or 1)
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Calculate object-aware similarity
    fg = gt_mask.mean()
    if fg == 0:
        obj_similarity = 1 - pred_mask.mean()  # All background
    elif fg == 1:
        obj_similarity = pred_mask.mean()      # All foreground
    else:
        o_fg = obj_foreground_similarity(pred_mask, gt_mask)
        o_bg = obj_background_similarity(pred_mask, gt_mask)
        obj_similarity = fg * o_fg + (1 - fg) * o_bg
    
    # Calculate region-aware similarity
    region_similarity = region_similarity_measure(pred_mask, gt_mask)
    
    # Calculate S-measure
    s_measure = alpha * obj_similarity + (1 - alpha) * region_similarity
    
    return s_measure

def obj_foreground_similarity(pred_mask, gt_mask):
    """Helper function for object-foreground similarity in S-measure"""
    # Get foreground areas
    fg_pred = pred_mask * gt_mask
    
    # Get means
    x_fg_mean = fg_pred.sum() / (gt_mask.sum() + 1e-8)
    
    # Calculate similarity
    if x_fg_mean == 0:
        return 0
    
    return 2 * x_fg_mean / (x_fg_mean + 1)

def obj_background_similarity(pred_mask, gt_mask):
    """Helper function for object-background similarity in S-measure"""
    # Get background areas
    bg_gt = 1 - gt_mask
    bg_pred = (1 - pred_mask) * bg_gt
    
    # Get means
    x_bg_mean = bg_pred.sum() / (bg_gt.sum() + 1e-8)
    
    # Calculate similarity
    if x_bg_mean == 0:
        return 0
    
    return 2 * x_bg_mean / (x_bg_mean + 1)

def region_similarity_measure(pred_mask, gt_mask):
    """Helper function for region similarity in S-measure"""
    # Apply Gaussian filter to smooth the masks
    pred_mask_smooth = gaussian_filter(pred_mask, sigma=1.0)
    gt_mask_smooth = gaussian_filter(gt_mask, sigma=1.0)
    
    # Calculate means
    mu_x = pred_mask_smooth.mean()
    mu_y = gt_mask_smooth.mean()
    
    # Calculate variances and covariance
    sig_x = np.sum((pred_mask_smooth - mu_x) ** 2) / (pred_mask_smooth.size - 1 + 1e-8)
    sig_y = np.sum((gt_mask_smooth - mu_y) ** 2) / (gt_mask_smooth.size - 1 + 1e-8)
    sig_xy = np.sum((pred_mask_smooth - mu_x) * (gt_mask_smooth - mu_y)) / (pred_mask_smooth.size - 1 + 1e-8)
    
    # Constants for numerical stability (as in SSIM)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Calculate region similarity (similar to SSIM)
    num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    
    return num / den if den > 0 else 0

def calculate_mae(pred_mask, gt_mask):
    """
    Calculate Mean Absolute Error between prediction and ground truth masks
    Args:
        pred_mask: predicted mask (normalized to [0, 1])
        gt_mask: ground truth mask (normalized to [0, 1])
    Returns:
        mae: mean absolute error
    """
    # Ensure masks are normalized to [0, 1]
    pred_mask = pred_mask.astype(np.float32) / 255.0 if pred_mask.max() > 1 else pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32) / 255.0 if gt_mask.max() > 1 else gt_mask.astype(np.float32)
    
    # Calculate absolute error
    mae = np.mean(np.abs(pred_mask - gt_mask))
    
    return mae

def masks_from_boxes(boxes, shape):
    """
    Convert bounding boxes to binary masks
    Args:
        boxes: tensor of boxes with shape [N, 4] in format [x1, y1, x2, y2]
        shape: output mask shape (height, width)
    Returns:
        masks: binary mask with shape specified by shape
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Clip to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    
    return mask

def evaluate_segmentation_metrics(model, dataloader):
    """
    Evaluate segmentation metrics including F-measure, S-measure, and MAE
    Args:
        model: trained model
        dataloader: validation dataloader
    Returns:
        Dictionary with segmentation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    f_measures = []
    s_measures = []
    maes = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating segmentation metrics'):
            images = batch['image'].to(device)
            masks = batch['mask']
            
            # Get predictions
            outputs = model(images)
            pred_boxes = outputs['proposals']
            
            # Calculate metrics for each image
            for i in range(images.shape[0]):
                gt_mask = masks[i][0].numpy()  # Get ground truth mask
                
                # Convert predicted boxes to mask
                h, w = gt_mask.shape
                if i < len(pred_boxes):
                    pred_boxes_np = pred_boxes[i].cpu().numpy()
                    pred_mask = masks_from_boxes(pred_boxes_np, (h, w))
                else:
                    pred_mask = np.zeros((h, w), dtype=np.uint8)
                
                # Calculate metrics
                _, _, f_measure = calculate_precision_recall_f_measure(pred_mask, gt_mask, beta=1)
                s_measure = calculate_s_measure(pred_mask, gt_mask)
                mae = calculate_mae(pred_mask, gt_mask)
                
                f_measures.append(f_measure)
                s_measures.append(s_measure)
                maes.append(mae)
    
    # Calculate average metrics
    avg_f_measure = np.mean(f_measures)
    avg_s_measure = np.mean(s_measures)
    avg_mae = np.mean(maes)
    
    metrics = {
        'F-measure': avg_f_measure,
        'S-measure': avg_s_measure,
        'MAE': avg_mae
    }
    
    return metrics

def evaluate_model(model, dataloader, iou_threshold=0.5):
    """
    Evaluate model performance using IoU and segmentation metrics
    Args:
        model: trained model
        dataloader: validation dataloader
        iou_threshold: threshold for positive detection
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_gt_boxes = 0
    total_pred_boxes = 0
    true_positives = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating bounding box metrics'):
            images = batch['image'].to(device)
            gt_boxes = batch['boxes']
            
            # Get predictions
            outputs = model(images)
            pred_boxes = outputs['proposals']
            
            # Calculate metrics for each image
            for i in range(images.shape[0]):
                gt_box = gt_boxes[i].cpu().numpy()
                pred_box = pred_boxes[i].cpu().numpy() if i < len(pred_boxes) else np.array([])
                
                total_gt_boxes += len(gt_box)
                total_pred_boxes += len(pred_box)
                
                # For each ground truth box, find the best matching prediction
                for gt in gt_box:
                    best_iou = 0
                    for pred in pred_box:
                        iou = calculate_iou(gt, pred)
                        best_iou = max(best_iou, iou)
                    
                    if best_iou >= iou_threshold:
                        true_positives += 1
    
    # Calculate precision and recall
    precision = true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate segmentation metrics
    print("Calculating segmentation metrics...")
    seg_metrics = evaluate_segmentation_metrics(model, dataloader)
    
    # Combine all metrics
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'F-measure': seg_metrics['F-measure'],
        'S-measure': seg_metrics['S-measure'],
        'MAE': seg_metrics['MAE']
    }
    
    return metrics 