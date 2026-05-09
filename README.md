# Deepfake Detection — Unified Slim Inception V1 + V2 + V3

A custom-built, from-scratch deep learning model for binary deepfake/real image classification. The model unifies three generations of Inception architecture (V1, V2, V3) into a single slim network with multi-scale skip fusion, trained on ~190K images with **90.71% test accuracy** and **0.9628 AUC**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture Deep Dive](#architecture-deep-dive)
  - [Stem](#stem)
  - [Stage 1 — Inception V1 Modules](#stage-1--inception-v1-modules)
  - [Stage 2 — Inception V2 Modules](#stage-2--inception-v2-modules)
  - [Stage 3 — Inception V3 Modules](#stage-3--inception-v3-modules)
  - [Multi-Scale Fusion Head](#multi-scale-fusion-head)
- [Model Summary](#model-summary)
- [Training](#training)
  - [Data Augmentation](#data-augmentation)
  - [Optimizer & Loss](#optimizer--loss)
  - [Callbacks](#callbacks)
  - [Training History (Epoch-by-Epoch)](#training-history-epoch-by-epoch)
- [Evaluation Results](#evaluation-results)
- [File Outputs](#file-outputs)
- [Environment](#environment)
- [How to Run](#how-to-run)

---

## Project Overview

This project builds a **custom unified Inception architecture** from scratch (no pretrained weights) for detecting AI-generated (deepfake) faces versus real images. Rather than using a single Inception generation, the model stacks V1, V2, and V3 modules sequentially with skip connections that feed earlier feature maps directly into the classification head — enabling multi-scale reasoning.

The design is intentionally "slim": filter counts are approximately **50% of original GoogLeNet/Inception-v3** sizes to reduce compute and memory while maintaining strong performance.

---

## Dataset

**Source:** [Kaggle — Deepfake and Real Images (manjilkarki)](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

| Split      | Samples | Path                    |
|------------|---------|-------------------------|
| Train      | 140,002 | `Dataset/Train/`        |
| Validation | 39,428  | `Dataset/Validation/`   |
| Test       | 10,905  | `Dataset/Test/`         |

**Classes:** `Fake` (index 0) and `Real` (index 1) — balanced binary classification.

**Image size:** 224 × 224 × 3 (RGB, rescaled to `[0, 1]`)

---

## Architecture Deep Dive

```
Input (224×224×3)
       │
  ┌────▼────┐
  │  STEM   │  Conv(32,3×3,s2) → BN → Conv(32,3×3) → BN → MaxPool(s2) → Conv(96,3×3) → BN → MaxPool(s2)
  └────┬────┘           Output: 28×28×96
       │
  ┌────▼──────────────────────┐
  │  STAGE 1: Inception V1    │  2 × V1 Module (parallel branches, no BN)
  │  v1_m1 → v1_m2 → MaxPool │  Output after pool: 14×14×240
  └────┬──────────────┬───────┘
       │              │
       │        skip_v1_gap → Dense(64)  ← [multi-scale skip #1]
       │
  ┌────▼──────────────────────┐
  │  STAGE 2: Inception V2    │  2 × V2 Module (5×5 replaced by 2×3×3 + BatchNorm)
  │  v2_m1 → v2_m2 → MaxPool │  Output after pool: 7×7×256
  └────┬──────────────┬───────┘
       │              │
       │        skip_v2_gap → Dense(64)  ← [multi-scale skip #2]
       │
  ┌────▼──────────────────────┐
  │  STAGE 3: Inception V3    │  3 × V3 Module (asymmetric 1×n / n×1 factorization + BN)
  │  v3_m1 → v3_m2 → v3_m3   │  Output: 7×7×416
  └────┬──────────────────────┘
       │
  GlobalAvgPool → Dense(128) → BN
       │
  Concatenate [128 + 64 + 64 = 256]  ← fuses all three scales
       │
  Dense(128, relu) → Dropout(0.5)
       │
  Dense(64, relu) → Dropout(0.3)
       │
  Dense(1, sigmoid)  →  P(Real)
```

### Stem

The stem aggressively downsamples the input from 224×224 to 28×28 using two Conv layers and two MaxPool layers, then expands to 96 channels.

| Layer       | Output Shape     | Params |
|-------------|------------------|--------|
| Conv(32,3×3,s2) + BN | 112×112×32 | 896 + 128 |
| Conv(32,3×3) + BN | 112×112×32 | 9,248 + 128 |
| MaxPool(s2) | 56×56×32 | — |
| Conv(96,3×3) + BN | 56×56×96 | 27,744 + 384 |
| MaxPool(s2) | 28×28×96 | — |

### Stage 1 — Inception V1 Modules

Classic GoogLeNet-style 4-branch parallel modules (no BatchNorm — original style):

- **Branch 1:** 1×1 Conv (dimensionality reduction / point-wise features)
- **Branch 2:** 1×1 Conv → 3×3 Conv (medium spatial features)
- **Branch 3:** 1×1 Conv → 5×5 Conv (wide spatial features)
- **Branch 4:** 3×3 MaxPool → 1×1 Conv (pooling branch)

All branches are concatenated along the channel axis.

| Module | Filter config (b1, b2r, b2, b3r, b3, bpool) | Output |
|--------|----------------------------------------------|--------|
| v1_m1  | 32, 48, 64, 8, 16, 16 | 28×28×128 |
| v1_m2  | 64, 64, 96, 16, 48, 32 | 28×28×240 |

After MaxPool(s2): **14×14×240** → skip connection tapped here.

### Stage 2 — Inception V2 Modules

The 5×5 convolution is replaced by **two stacked 3×3 convolutions** (same receptive field, fewer parameters). BatchNormalization is added after every conv.

AveragePooling replaces MaxPooling in the pooling branch (V2 style).

| Module | Output |
|--------|--------|
| v2_m1  | 14×14×256 |
| v2_m2  | 14×14×256 |

After MaxPool(s2): **7×7×256** → skip connection tapped here.

### Stage 3 — Inception V3 Modules

Each conv is factorized into asymmetric **1×n** then **n×1** convolutions (computationally cheaper than n×n). The deeper branch applies this factorization twice.

| Module | n (factorization) | Output |
|--------|-------------------|--------|
| v3_m1  | 3 | 7×7×256 |
| v3_m2  | 3 | 7×7×256 |
| v3_m3  | 5 | 7×7×416 |

### Multi-Scale Fusion Head

After Stage 3, a GlobalAveragePool collapses 7×7×416 to a 416-dim vector, then a Dense(128)+BN projects it down to 128 dims.

This is **concatenated** with the two skip connection outputs (64 each), giving a **256-dim multi-scale feature vector** that sees coarse low-level information from V1 as well as the deep abstract features from V3.

```
final_gap(416) → Dense(128) → BN → [128-dim]
skip_v1_fc  →                       [64-dim]   ──► Concat(256) → Dense(128) → Dropout(0.5)
skip_v2_fc  →                       [64-dim]                      → Dense(64) → Dropout(0.3)
                                                                   → Dense(1, sigmoid)
```

---

## Model Summary

| Property             | Value             |
|----------------------|-------------------|
| Model name           | `Unified_InceptionV1_V2_V3_Slim` |
| Total parameters     | **1,070,473** (~4.08 MB) |
| Trainable params     | 1,066,217 |
| Non-trainable params | 4,256 |
| Input shape          | (224, 224, 3) |
| Output               | Sigmoid (binary) |

---

## Training

### Data Augmentation

Applied only to training data. Validation and test use rescaling only.

| Augmentation       | Value     |
|--------------------|-----------|
| Rescale            | 1/255     |
| Rotation range     | ±15°      |
| Width shift        | 10%       |
| Height shift       | 10%       |
| Zoom range         | 15%       |
| Horizontal flip    | Yes       |
| Fill mode          | nearest   |

### Optimizer & Loss

| Setting            | Value                        |
|--------------------|------------------------------|
| Optimizer          | Adam (lr = 1e-3)             |
| Loss               | Binary Crossentropy          |
| Metrics            | Accuracy, AUC, Precision, Recall |
| Batch size         | 32                           |
| Max epochs         | 30                           |

### Callbacks

| Callback           | Config                                              |
|--------------------|-----------------------------------------------------|
| ModelCheckpoint    | Monitor `val_accuracy`, save best only              |
| EarlyStopping      | Monitor `val_accuracy`, patience=8, restore best weights |
| ReduceLROnPlateau  | Monitor `val_loss`, factor=0.3, patience=3, min_lr=1e-7 |

### Training History (Epoch-by-Epoch)

| Epoch | Train Acc | Train AUC | Val Acc | Val AUC | Val Loss | LR     | Saved? |
|-------|-----------|-----------|---------|---------|----------|--------|--------|
| 1     | 0.7433    | 0.8211    | 0.9073  | 0.9731  | 0.2350   | 0.0010 | ✅     |
| 2     | 0.9354    | 0.9823    | 0.9288  | 0.9804  | 0.1869   | 0.0010 | ✅     |
| 3     | 0.9532    | 0.9896    | 0.9308  | 0.9822  | 0.1830   | 0.0010 | ✅     |
| 4     | 0.9586    | 0.9919    | 0.9392  | 0.9859  | 0.1565   | 0.0010 | ✅     |
| 5     | 0.9627    | 0.9931    | 0.8841  | 0.9769  | 0.3342   | 0.0010 | ❌     |
| 6     | 0.9659    | 0.9942    | 0.8877  | 0.9644  | 0.2625   | 0.0010 | ❌     |
| 7     | 0.9677    | 0.9946    | 0.9604  | 0.9918  | 0.1215   | 0.0010 | ✅ (best) |
| 8     | 0.9688    | 0.9950    | 0.9228  | 0.9777  | 0.2513   | 0.0010 | ❌     |
| 9     | 0.9710    | 0.9957    | 0.9554  | 0.9921  | 0.1174   | 0.0010 | ❌     |
| 10    | 0.9741    | 0.9961    | 0.9444  | 0.9880  | 0.1688   | 0.0010 | ❌     |
| 11    | 0.9726    | 0.9960    | 0.9152  | 0.9793  | 0.2586   | 0.0010 | ❌     |
| 12    | 0.9738    | 0.9962    | 0.9569  | 0.9935  | 0.1066   | 0.0010 | ❌     |
| 13    | 0.9748    | 0.9967    | 0.9568  | 0.9913  | 0.1215   | 0.0010 | ❌     |
| 14    | 0.9761    | 0.9968    | 0.9604  | 0.9928  | 0.1037   | 0.0010 | ❌     |
| 15    | 0.9758    | 0.9967    | 0.9370  | 0.9867  | 0.1670   | 0.0010 | ❌ (early stop) |

**Training stopped at epoch 15** (patience=8 from epoch 7 best). Best model weights from **epoch 7** were restored.

---

## Evaluation Results

Evaluated on the held-out test set of **10,905 images** (5,492 Fake / 5,413 Real).

### Overall Metrics

| Metric         | Score  |
|----------------|--------|
| Test Loss      | 0.2903 |
| Test Accuracy  | **90.71%** |
| Test AUC       | **0.9628** |
| Test Precision | 0.9324 |
| Test Recall    | 0.8764 |

### Classification Report

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Fake     | 0.88      | 0.94   | **0.91** | 5,492   |
| Real     | 0.93      | 0.88   | **0.90** | 5,413   |
| Accuracy |           |        | **0.91** | 10,905  |
| Macro avg| 0.91      | 0.91   | 0.91     | 10,905  |

The model is slightly better at **recalling fakes** (94%) but has higher **precision on real images** (93%), indicating a mild bias toward flagging images as fake.

---

## File Outputs

| File                              | Description                                    |
|-----------------------------------|------------------------------------------------|
| `unified_inception_slim_best.keras` | Best checkpoint (by `val_accuracy`, epoch 7) |
| `unified_inception_slim.keras`    | Final model after training completes           |
| `training_history.png`            | 2×2 plot of Accuracy, Loss, AUC, Precision     |
| `confusion_matrix.png`            | Confusion matrix heatmap on test set           |

---

## Environment

| Component     | Version / Detail                      |
|---------------|---------------------------------------|
| TensorFlow    | 2.19.0                                |
| Python        | 3.x (Kaggle kernel)                   |
| GPU           | 2 × Tesla T4 (13,757 MB each)         |
| CUDA          | cuDNN 9.1.0.2                         |
| Platform      | Kaggle Notebooks                      |

---

## How to Run

### 1. Set up the dataset

Download the dataset from Kaggle and set `DATASET_PATH` to its location:

```python
DATASET_PATH = '/kaggle/input/datasets/manjilkarki/deepfake-and-real-images/Dataset'
```

Or update to your local path:

```python
DATASET_PATH = '/your/local/path/Dataset'
```

### 2. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run the notebook

Open `Deepfake_detection_inception.ipynb` in Jupyter or Kaggle and run all cells sequentially. The single notebook cell handles:

- Data loading and augmentation
- Model building
- Training with callbacks
- Evaluation and metrics
- Saving the model

### 4. Inference on a new image

```python
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image

model = keras.models.load_model('unified_inception_slim_best.keras')

img = image.load_img('your_image.jpg', target_size=(224, 224))
x = image.img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

prob = model.predict(x)[0][0]
label = "Real" if prob > 0.5 else "Fake"
print(f"Prediction: {label} (confidence: {prob:.3f})")
```

---

## Key Design Decisions

- **No pretrained weights** — the model is trained from scratch on the deepfake dataset, avoiding domain mismatch with ImageNet.
- **Slim filter counts (~50% of originals)** — reduces compute by ~40% while maintaining good accuracy for this domain.
- **Multi-scale skip connections** — instead of discarding intermediate feature maps, V1 and V2 outputs are tapped via GlobalAveragePool and projected to 64-dim vectors, then concatenated with the final V3 representation. This lets the classifier see both low-level texture artifacts and high-level semantic features simultaneously.
- **Asymmetric factorization in V3** — 1×n + n×1 convolutions are computationally cheaper than n×n while covering the same receptive field.
- **AveragePooling in V2/V3 branches** — following the original papers, average pooling better preserves spatial information in the pooling branch compared to max pooling.
