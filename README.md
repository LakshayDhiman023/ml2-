# Video Object Segmentation with ResNet50 Backbone

This project implements a video object segmentation model using the DAVIS dataset with a ResNet50 backbone. The model uses region proposals, ROI pooling, and fully connected layers for semantic segmentation.

## Features

- Loads and preprocesses the DAVIS dataset
- Implements a ResNet50 backbone with custom feature extraction
- Adds convolutional layers for region proposals
- Uses Non-Maximum Suppression (NMS) for proposal pruning
- Implements ROI pooling from torchvision.ops
- Adds fully connected layers for classification
- Training loop with loss tracking
- Visualization of predictions
- Comprehensive evaluation metrics including:
  - Precision, Recall, F1 score for bounding boxes
  - F-Measure for segmentation masks
  - S-Measure (structure measure) for segmentation quality
  - MAE (Mean Absolute Error) between prediction and ground truth

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The code expects the DAVIS dataset in the following structure:

```
DAVIS-2017-trainval-480p/
└── DAVIS/
    ├── JPEGImages/
    │   └── 480p/
    │       └── [sequence_folders]/
    ├── Annotations/
    │   └── 480p/
    │       └── [sequence_folders]/
    └── ImageSets/
        └── 2017/
            ├── train.txt
            └── val.txt
```

## Usage

### Testing on a Small Subset

Before running the full training, you can test on a small subset:

```bash
python test_small_subset.py
```

### Testing Segmentation Metrics

To test the segmentation metrics (F-Measure, S-Measure, MAE):

```bash
python test_metrics.py
```

### Training

To train the model:

```bash
python train.py
```

This will:
1. Load the DAVIS dataset
2. Initialize the model with a pretrained ResNet50 backbone
3. Train for 10 epochs with a learning rate of 1e-4
4. Save checkpoints in the `checkpoints` directory
5. Visualize predictions in the `predictions` directory
6. Calculate and report comprehensive evaluation metrics

### Model Architecture

- **Backbone**: Modified ResNet50 with intermediate feature maps
- **Proposal Network**: Three convolutional layers (3x3, 1x1, 1x1) with ReLU and batch normalization
- **Proposal Pruning**: Non-Maximum Suppression (NMS)
- **ROI Pooling**: Fixed output size of 7x7
- **FC Layers**: Two fully connected layers with 4096 units each
- **Output**: Classification and bounding box regression

## Evaluation Metrics

### Bounding Box Metrics
- **Precision**: Ratio of correctly predicted boxes to total predicted boxes
- **Recall**: Ratio of correctly predicted boxes to total ground truth boxes
- **F1 Score**: Harmonic mean of precision and recall

### Segmentation Metrics
- **F-Measure**: Combines precision and recall with β=1 for segmentation masks
- **S-Measure**: Measures structure-aware quality by combining region and object similarity
- **MAE**: Mean Absolute Error between predicted and ground truth masks

## Results

After training, you'll find:
- Model checkpoints in the `checkpoints` directory
- A plot of training and validation loss
- Visualizations of predictions in the `predictions` directory
- Comprehensive evaluation metrics in the console output #   m l 2 -  
 