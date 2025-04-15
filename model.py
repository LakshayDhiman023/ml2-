import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import roi_pool, nms

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        # Load ResNet50 with proper weights parameter instead of deprecated pretrained
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        model = resnet50(weights=weights)
        
        # Remove final layers (avg pool, fc)
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        
        # Get output feature map size - should be [2048, 7, 7] for a 224x224 input
        # We'll add a 1x1 conv to reduce channels from 2048 -> 512
        self.channel_reducer = nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Reduce channels
        features = self.channel_reducer(features)
        features = self.bn(features)
        features = self.relu(features)
        
        return features

class ProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(ProposalNetwork, self).__init__()
        
        # 3x3 conv
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 1x1 conv
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Final 1x1 conv for proposal prediction
        # 4 for bounding box coordinates, 1 for objectness score
        self.proposal_conv = nn.Conv2d(512, 5, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        proposals = self.proposal_conv(x)
        
        return proposals

class ProposalProcessor(nn.Module):
    def __init__(self, nms_threshold=0.7, score_threshold=0.5):
        super(ProposalProcessor, self).__init__()
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
    
    def forward(self, proposals, image_shape):
        """
        Process proposals using NMS
        Args:
            proposals: tensor of shape [B, 5, H, W] with bbox coordinates and confidence score
            image_shape: original image shape (height, width)
        Returns:
            pruned_boxes: list of tensors with pruned boxes for each batch item
        """
        batch_size = proposals.shape[0]
        pruned_boxes = []
        
        for b in range(batch_size):
            # Get proposal map for this batch item
            # Shape: [5, H, W]
            prop_map = proposals[b]
            
            # Convert from feature map to bounding boxes
            # Get height and width of feature map
            h, w = prop_map.shape[1:]
            
            # Create a grid of coordinates
            y_grid, x_grid = torch.meshgrid(torch.arange(h, device=prop_map.device),
                                          torch.arange(w, device=prop_map.device), indexing='ij')
            
            # Reshape to [H*W, 2]
            grid = torch.stack([x_grid, y_grid], dim=2).reshape(-1, 2).float()
            
            # Extract the 5 channels
            # First 4 are the bbox coordinates (relative to cell)
            # Last one is objectness score
            prop_map_reshaped = prop_map.permute(1, 2, 0).reshape(-1, 5)
            
            # Convert relative coordinates to absolute
            # Proposals encode: center_x, center_y, width, height
            cx = grid[:, 0] + prop_map_reshaped[:, 0]  # add relative center_x offset
            cy = grid[:, 1] + prop_map_reshaped[:, 1]  # add relative center_y offset
            w_boxes = torch.exp(prop_map_reshaped[:, 2])     # width
            h_boxes = torch.exp(prop_map_reshaped[:, 3])     # height
            
            # Convert to [x1, y1, x2, y2] format
            boxes = torch.zeros_like(prop_map_reshaped[:, :4])
            boxes[:, 0] = cx - w_boxes/2  # x1
            boxes[:, 1] = cy - h_boxes/2  # y1
            boxes[:, 2] = cx + w_boxes/2  # x2
            boxes[:, 3] = cy + h_boxes/2  # y2
            
            # Get scores
            scores = torch.sigmoid(prop_map_reshaped[:, 4])
            
            # Filter by score threshold
            mask = scores > self.score_threshold
            filtered_boxes = boxes[mask]
            filtered_scores = scores[mask]
            
            # If no boxes pass the threshold, take the top scoring one
            if filtered_boxes.shape[0] == 0:
                top_idx = torch.argmax(scores)
                filtered_boxes = boxes[top_idx:top_idx+1]
                filtered_scores = scores[top_idx:top_idx+1]
            
            # Apply NMS
            keep_indices = nms(filtered_boxes, filtered_scores, self.nms_threshold)
            
            # Get final pruned boxes
            pruned = filtered_boxes[keep_indices]
            
            # Scale boxes to original image size
            # The feature map dimensions are h and w from above
            img_h, img_w = image_shape
            
            # Use scalars to avoid broadcasting issues
            scale_w = img_w / w
            scale_h = img_h / h
            
            # Scale the boxes
            scaled_pruned = pruned.clone()
            scaled_pruned[:, 0] = pruned[:, 0] * scale_w
            scaled_pruned[:, 1] = pruned[:, 1] * scale_h
            scaled_pruned[:, 2] = pruned[:, 2] * scale_w
            scaled_pruned[:, 3] = pruned[:, 3] * scale_h
            
            pruned_boxes.append(scaled_pruned)
        
        return pruned_boxes

class ROIHead(nn.Module):
    def __init__(self, roi_size=7, num_classes=1):
        super(ROIHead, self).__init__()
        
        # FC layers after ROI pooling
        self.fc1 = nn.Linear(512 * roi_size * roi_size, 4096)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # Output layers
        self.cls_score = nn.Linear(4096, num_classes + 1)  # +1 for background
        self.bbox_pred = nn.Linear(4096, 4 * num_classes)  # 4 coordinates per class
        
        self.roi_size = roi_size
    
    def forward(self, features, rois, batch_indices):
        """
        Forward pass through ROI head
        Args:
            features: feature map from backbone [B, C, H, W]
            rois: list of roi boxes, each with shape [N, 4]
            batch_indices: list of batch indices for each roi
        """
        # Stack all ROIs and create batch index tensor
        if len(rois) == 0:
            return None, None
        
        rois_tensor = torch.cat(rois, dim=0)
        batch_idx_tensor = torch.cat([torch.full((r.shape[0],), i, dtype=torch.int64, device=features.device) 
                                    for i, r in enumerate(rois)])
        
        # Apply ROI pooling
        # Format: [N, C, roi_size, roi_size]
        roi_features = roi_pool(features, 
                               torch.cat([batch_idx_tensor.unsqueeze(1), rois_tensor], dim=1),
                               output_size=(self.roi_size, self.roi_size),
                               spatial_scale=1.0/16.0)  # 1/16 for ResNet downsampling
        
        # Flatten
        roi_features = roi_features.view(roi_features.shape[0], -1)
        
        # Apply FC layers
        x = self.fc1(roi_features)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Get classification and regression outputs
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        
        return cls_scores, bbox_preds

class VideoSegmentationModel(nn.Module):
    def __init__(self, num_classes=1, pretrained_backbone=True, roi_size=7, nms_threshold=0.7):
        super(VideoSegmentationModel, self).__init__()
        
        # Backbone
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        
        # Proposal network
        self.proposal_net = ProposalNetwork(in_channels=512)
        
        # Proposal processor with NMS
        self.proposal_processor = ProposalProcessor(nms_threshold=nms_threshold)
        
        # ROI Head
        self.roi_head = ROIHead(roi_size=roi_size, num_classes=num_classes)
    
    def forward(self, x):
        """
        Forward pass through the full model
        Args:
            x: input tensor of shape [B, C, H, W]
        Returns:
            Dictionary with detection results
        """
        batch_size = x.shape[0]
        original_size = (x.shape[2], x.shape[3])  # (H, W)
        
        # Extract features from backbone
        features = self.backbone(x)
        
        # Generate proposals
        proposal_maps = self.proposal_net(features)
        
        # Process proposals with NMS
        # Returns a list of tensors, each with shape [N, 4] where N is variable
        pruned_proposals = self.proposal_processor(proposal_maps, original_size)
        
        # Create batch indices for ROI pooling
        batch_indices = [torch.full((boxes.shape[0],), i, dtype=torch.int64, device=x.device) 
                        for i, boxes in enumerate(pruned_proposals)]
        
        # Pass through ROI head
        cls_scores, bbox_preds = self.roi_head(features, pruned_proposals, batch_indices)
        
        result = {
            'features': features,
            'proposals': pruned_proposals,
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }
        
        return result 