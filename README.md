# Image Classification Comparison: Traditional ML vs Deep Learning

A comprehensive Jupyter notebook comparing multiple image classification approaches across traditional machine learning and modern deep learning frameworks.

## 📋 Overview

This notebook demonstrates end-to-end image classification pipelines using:

- **Traditional ML**: Support Vector Machines (SVM) with GridSearchCV, Random Forest with HOG features
- **Deep Learning**: CNN (Sequential Keras), Advanced CNN (Functional API with augmentation), PyTorch Lightning CNN

All models are trained and evaluated on popular datasets (CIFAR-10, CIFAR-100, Cats vs Dogs) with comprehensive metrics, confusion matrices, and visualizations.

## 🚀 Features

- **Dataset Integration**: CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, Cats vs Dogs (via TensorFlow Datasets)
- **Multi-Framework**: TensorFlow/Keras, scikit-learn, PyTorch, PyTorch Lightning
- **Optimized Training**: Early stopping, learning rate scheduling, data augmentation, mixed precision (GPU)
- **Rich Visualizations**: Training curves, confusion matrices, sample predictions with confidence scores
- **Detailed Metrics**: Accuracy, precision, recall, F1-score, classification reports

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Install Dependencies

```bash
pip install tensorflow keras torch torchvision pytorch-lightning torchmetrics \
            scikit-learn scikit-image matplotlib numpy pandas \
            tensorflow-datasets pillow
```

Or using conda:

```bash
conda create -n image-classification python=3.10
conda activate image-classification
pip install -r requirements.txt  # (if provided)
```

## 🎯 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jugalmodi0111/image_classification_comparison.git
   cd image_classification_comparison
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook image_classification_comparison.ipynb
   ```

3. **Run cells sequentially** or run all cells to execute complete pipelines.

## 📊 Models & Performance

### 1. **SVM Classifier** (Cats vs Dogs)
- Features: Flattened 64×64 RGB images
- Optimization: GridSearchCV with reduced parameter grid
- Training time: ~2-3 minutes (500 samples, CPU)

### 2. **Random Forest + HOG** (Cats vs Dogs)
- Features: Histogram of Oriented Gradients (HOG) from 128×128 grayscale images
- Training time: ~1-2 minutes (800 train / 200 test samples)

### 3. **CNN Sequential** (CIFAR-10)
- Architecture: 3 Conv blocks + Dropout + Dense layers
- Training: 15 epochs with validation split
- Typical accuracy: ~70-75%

### 4. **Advanced Functional CNN** (CIFAR-10) **[OPTIMIZED]**
- Architecture: Deeper network with BatchNormalization
- **Optimizations**:
  - In-model data augmentation (RandomFlip, RandomTranslation, RandomRotation)
  - EarlyStopping (patience=6) + ReduceLROnPlateau
  - Reduced Dense layer (256 vs 1024) for faster convergence
  - tf.data pipeline with prefetching
  - Optional mixed precision for GPU
- Training: Up to 35 epochs (early stopping typically triggers ~20-25)
- **Speed improvement**: 2-3× faster than original dual-phase training
- Typical accuracy: ~75-80%

### 5. **PyTorch Lightning CNN** (CIFAR-10)
- Architecture: 3 Conv layers + FC layers
- Features: Automatic logging, checkpointing, test evaluation
- Training: 5 epochs (CPU, quick demo)

## 🔧 Optimization Notes (TensorFlow Cell)

The original TensorFlow functional model ran for **100 epochs total** (50 without augmentation + 50 with ImageDataGenerator), which was very slow.

**Key Optimizations Applied**:
- ✅ **Single training phase** with augmentation layers inside the model graph
- ✅ **EarlyStopping** (monitor `val_accuracy`, patience=6, restore best weights)
- ✅ **ReduceLROnPlateau** (factor=0.5, patience=3)
- ✅ **ModelCheckpoint** (saves `best_cifar_model.keras`)
- ✅ **Smaller Dense layer** (1024 → 256 neurons)
- ✅ **tf.data pipeline** with `.shuffle().batch().prefetch()` for better throughput
- ✅ **Mixed precision** (optional, GPU-only)

**Result**: Training completes in **~20-25 epochs** instead of 100, with comparable or better accuracy.

## 📁 Project Structure

```
.
├── image_classification_comparison.ipynb  # Main notebook
├── README.md                              # This file
├── .gitignore                             # Git ignore rules
├── best_cifar_model.keras                 # Saved TensorFlow model (auto-generated)
├── image_classifier.ckpt                  # PyTorch Lightning checkpoint (auto-generated)
└── data/                                  # Downloaded datasets (auto-created)
```

## 🛠️ Usage Tips

- **For quick testing**: Run SVM and Random Forest cells (fast, <5 min total)
- **For deep learning demos**: Run CNN Sequential (moderate, ~10-15 min)
- **For optimized training**: Run Advanced Functional CNN (best performance/time ratio)
- **For PyTorch users**: Run Lightning cell (minimal example, 5 epochs)

## 📈 Metrics & Visualization

All models include:
- **Classification reports** (per-class precision, recall, F1)
- **Confusion matrices** (heatmaps with actual vs predicted)
- **Training curves** (accuracy & loss over epochs)
- **Sample predictions** (visual grid with confidence scores)

## 🤝 Contributing

Contributions welcome! Potential improvements:
- Add transfer learning (ResNet50, EfficientNet)
- Implement additional datasets (ImageNet subsets, custom data)
- Add model export (ONNX, TensorFlow Lite)
- Include hyperparameter tuning notebooks

## 📝 License

MIT License - feel free to use for educational or research purposes.

## 👤 Author

**Jugal Modi**
- GitHub: [@jugalmodi0111](https://github.com/jugalmodi0111)

## 🙏 Acknowledgments

- TensorFlow & Keras teams for excellent APIs
- PyTorch Lightning for simplified training loops
- scikit-learn for classical ML implementations
- TensorFlow Datasets for easy dataset access
