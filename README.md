
```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║     ██████╗  ██████╗  ██████╗██╗  ██╗██╗███╗   ██╗██╗   ║
║     ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██║████╗  ██║██║   ║
║     ██║  ██║██║   ██║██║     █████╔╝ ██║██╔██╗ ██║██║   ║
║     ██║  ██║██║   ██║██║     ██╔═██╗ ██║██║╚██╗██║██║   ║
║     ██████╔╝╚██████╔╝╚██████╗██║  ██╗██║██║ ╚████║██║   ║
║     ╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝   ║
║                                                          ║
║              🔐 PRECISION POSITION PREDICTOR 🔐          ║
║         Advanced ResNet50-Powered Docking System         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

# ResNet50 Docking Position Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A deep learning model for predicting docking positions using ResNet50. This system predicts the optimal x and y coordinates and distance for molecular/protein docking scenarios.

## 🎯 Overview

This project leverages a pre-trained ResNet50 architecture fine-tuned for regression tasks to predict docking positions from image data. The model outputs three continuous values: **x-coordinate**, **y-coordinate**, and **distance**, making it suitable for precision docking applications.

**Key Features:**
- ✅ **Pre-trained ResNet50** - Leverages ImageNet weights for transfer learning
- ✅ **Checkpoint Management** - Automatic resumable training with best model tracking
- ✅ **Early Stopping** - Prevents overfitting with configurable patience
- ✅ **Batch Prediction** - Efficient batch-based inference on test datasets
- ✅ **Visualization** - Generate detailed prediction visualizations with actual vs predicted overlays
- ✅ **GPU Support** - Full CUDA support for faster training and inference

## 📊 Project Structure

```
Docking/
├── train_resnet50.py              # Training script with checkpoint resumption
├── datapreprocess.py              # Data loading and preprocessing utilities
├── predict_and_visualize.py       # Inference and visualization tools
├── checkpoint_manager.py          # Checkpoint inspection and management
├── visualize_loader.py            # Dataset visualization utilities
│
├── Dataset/                       # Processed dataset
│   ├── train.csv                 # Training split metadata
│   ├── val.csv                   # Validation split metadata
│   ├── test.csv                  # Test split metadata
│   └── images/                   # Image files
│
├── raw_data/                     # Original unprocessed data
│   ├── train.csv
│   └── images/
│
├── Visualization/                # Output predictions and visualizations
│   └── predictions_detailed.csv  # Detailed prediction results
│
├── README.md                     # This file
├── QUICK_START.md               # Quick reference guide
├── training_checkpoint.pth      # Training checkpoint (auto-generated)
└── resnet50_docking_best.pth   # Best model weights (auto-generated)
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Docking
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision torchaudio
   pip install pandas numpy matplotlib pillow scikit-learn
   ```

3. **Prepare your dataset:**
   - Place images in `Dataset/images/`
   - Ensure CSV files contain columns: `image_name`, `x_norm`, `y_norm`, `distance_norm`

### Training

**Start training (automatically resumes from checkpoint if one exists):**
```bash
python train_resnet50.py
```

**Check training progress:**
```bash
python checkpoint_manager.py status
```

**Reset training and start fresh:**
```bash
python checkpoint_manager.py reset
```

### Prediction & Visualization

**Generate predictions on test set:**
```bash
python predict_and_visualize.py
```

This will:
- Load the best trained model
- Process all test images
- Generate prediction circles overlaid on images
- Save results to `Visualization/predictions_detailed.csv`

## 📋 Model Architecture

| Component | Details |
|-----------|---------|
| **Base Model** | ResNet50 (pretrained on ImageNet) |
| **Input Size** | 3 × 512 × 512 (RGB images) |
| **Output Size** | 3 (x-coordinate, y-coordinate, distance) |
| **Loss Function** | SmoothL1Loss (robust to outliers) |
| **Optimizer** | Adam (learning rate: 1e-4) |
| **Batch Size** | 32 |
| **Epochs** | 20 (with early stopping) |
| **Early Stop Patience** | 15 epochs |
| **Weight Decay** | 1e-5 (L2 regularization) |

## ⚙️ Configuration

Key training parameters can be adjusted in `train_resnet50.py`:

```python
BATCH_SIZE = 32              # Training batch size
EPOCHS = 20                  # Maximum epochs
LEARNING_RATE = 1e-4         # Adam learning rate
EARLY_STOP_PATIENCE = 15     # Early stopping threshold
L2_REGULARIZATION = 1e-5     # Weight decay for regularization
AUTO_RESUME = True           # Resume from checkpoint automatically
```

## 📈 Training Details

### Checkpoint System

The training system automatically manages two checkpoint files:

1. **`resnet50_docking_best.pth`** ⭐
   - Best model weights (lowest validation loss)
   - Used for inference
   - Updated only when validation loss improves

2. **`training_checkpoint.pth`** 💾
   - Complete training state
   - Includes model weights, optimizer state, epoch counter, and metrics
   - Updated after every epoch
   - Enables seamless training resumption

### Early Stopping

Training automatically stops if validation loss doesn't improve for 15 consecutive epochs, preventing overfitting and saving computation time.

## 🎨 Outputs

### Prediction Visualization

The visualization script generates:
- **Overlay Images**: Original images with predicted docking circles
- **CSV Results**: Detailed predictions including:
  - Image filename
  - Predicted coordinates (pixels and normalized)
  - Predicted distance
  - Model confidence metrics (if applicable)

## 📊 Dataset Format

CSV files should follow this structure:

```csv
image_name,x_norm,y_norm,distance_norm
img_001.jpg,0.45,0.52,0.30
img_002.jpg,0.38,0.61,0.25
...
```

**Note:** Values are normalized to [0, 1] range

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `BATCH_SIZE` in `train_resnet50.py` |
| Training too slow | Enable GPU via CUDA, or use mixed precision training |
| Model not improving | Reduce `LEARNING_RATE` or increase `EPOCHS` |
| Checkpoint corrupted | Run `python checkpoint_manager.py reset` to start fresh |

## 📝 Key Files Description

| File | Purpose |
|------|---------|
| `train_resnet50.py` | Main training loop with checkpoint management and early stopping |
| `datapreprocess.py` | Dataset class, data loaders, image preprocessing |
| `predict_and_visualize.py` | Inference engine and visualization generation |
| `checkpoint_manager.py` | Checkpoint status inspection and reset utilities |
| `visualize_loader.py` | Dataset visualization for exploration and debugging |

## 📚 Dependencies

- **PyTorch** (2.0+) - Deep learning framework
- **torchvision** - Computer vision utilities and pretrained models
- **pandas** - Data manipulation and CSV handling
- **NumPy** - Numerical computations
- **Pillow** - Image loading and manipulation
- **Matplotlib** - Visualization and plotting
- **scikit-learn** - Data splitting utilities

## 💡 Tips for Better Results

1. **Data Quality**: Ensure images are properly preprocessed and normalized
2. **Data Balance**: Keep roughly equal distribution across training/validation/test splits
3. **Hyperparameter Tuning**: Experiment with `LEARNING_RATE` and `EARLY_STOP_PATIENCE`
4. **Augmentation**: Consider adding data augmentation for improved generalization
5. **GPU Usage**: Utilize GPU for significantly faster training

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues, questions, or suggestions, please open an issue in the repository.

---

**Last Updated:** April 2026
