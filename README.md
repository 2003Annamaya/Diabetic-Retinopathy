# 👁️ Diabetic Retinopathy Detection using Deep Learning

An AI/ML project that uses **transfer learning** with **ResNet50 (PyTorch)** to classify retinal fundus images into **5 severity levels** of Diabetic Retinopathy (DR). Includes a **Flask web application** for real-time image upload and prediction.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [DR Severity Classes](#dr-severity-classes)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Results](#results)
- [Web Application](#web-application)
- [Technologies](#technologies)

---

## 🔍 Overview

Diabetic Retinopathy is a diabetes complication that damages the blood vessels of the retina and is a leading cause of blindness worldwide. Early detection through AI-powered screening can help prevent vision loss.

This project:
- Uses **ResNet50** backbone with **ImageNet V2** pre-trained weights
- Applies **two-phase transfer learning** (frozen backbone → fine-tuning)
- Trains on **35,126** retinal fundus images with data augmentation
- Provides a **dark-themed web interface** for uploading retinal images and viewing predictions
- Generates **evaluation reports** with confusion matrices, ROC curves, and classification reports

---

## 🏷️ DR Severity Classes

| Grade | Class          | Description                                    |
|-------|----------------|------------------------------------------------|
| 0     | No DR          | No diabetic retinopathy detected               |
| 1     | Mild           | Mild non-proliferative DR (microaneurysms)     |
| 2     | Moderate       | Moderate non-proliferative DR                  |
| 3     | Severe         | Severe non-proliferative DR                    |
| 4     | Proliferative  | Proliferative DR (most advanced stage)         |

---

## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
├── config.py                    # Configuration & hyperparameters
├── train.py                     # Two-phase training pipeline
├── evaluate.py                  # Model evaluation (metrics, plots)
├── predict.py                   # Single-image inference
├── app.py                       # Flask web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── models/
│   ├── __init__.py
│   └── retinopathy_model.py     # DRClassifier (ResNet50 + custom head)
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py         # Data transforms & loaders
│
├── templates/
│   └── index.html               # Web UI (dark theme)
│
├── static/
│   └── uploads/                 # Uploaded images (auto-created)
│
├── saved_models/                # Trained model weights (download separately)
│   ├── best_dr_model.pth
│   └── DR_ResNet50_Final_Model.pth
│
├── results/                     # Evaluation outputs
│   ├── training_summary.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── per_class_accuracy.png
│   ├── sample_predictions.png
│   └── classification_report.txt
│
├── data/                        # Dataset (not included — see below)
│   ├── train/
│   ├── val/
│   └── test/
│
├── DR_Detection_Notebook.ipynb  # Full training & evaluation notebook
└── PPT_Content.txt              # Presentation content
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU Support**: For CUDA-enabled PyTorch, install from [pytorch.org](https://pytorch.org/get-started/locally/) instead:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Download trained model weights

The `.pth` model files are too large for GitHub (100MB limit). Download them and place in `saved_models/`:

```
saved_models/
├── best_dr_model.pth            (~92 MB)
└── DR_ResNet50_Final_Model.pth  (~92 MB)
```

> � You can also train from scratch using `python train.py` if you have the dataset.

---

## �📂 Dataset Preparation

This project expects a folder-structured image dataset. You can use:

- [**Kaggle – Diabetic Retinopathy Detection**](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)
- [**APTOS 2019 Blindness Detection**](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

### Steps:
1. Download the dataset from Kaggle
2. Organize images into the folder structure:
   ```
   data/
   ├── train/
   │   ├── No_DR/          (class 0 images)
   │   ├── Mild/           (class 1 images)
   │   ├── Moderate/       (class 2 images)
   │   ├── Severe/         (class 3 images)
   │   └── Proliferative/  (class 4 images)
   ├── val/                (same sub-folders)
   └── test/               (same sub-folders)
   ```
3. Recommended split: **70% train / 15% val / 15% test**

### Dataset Statistics (used in this project):
| Split     | No_DR  | Mild  | Moderate | Severe | Proliferative | Total  |
|-----------|--------|-------|----------|--------|---------------|--------|
| Train     | 18,067 | 1,710 | 3,704    | 611    | 496           | 24,588 |
| Val       | 3,871  | 367   | 794      | 131    | 106           | 5,269  |
| Test      | 3,872  | 366   | 794      | 131    | 106           | 5,269  |
| **Total** | **25,810** | **2,443** | **5,292** | **873** | **708** | **35,126** |

---

## 🚀 Usage

### Train the Model
```bash
python train.py
```

### Evaluate on Test Set
```bash
python evaluate.py
```

### Predict on a Single Image
```bash
python predict.py path/to/retinal_image.jpg
```

### Launch Web Application
```bash
python app.py
# Open http://127.0.0.1:5000 in your browser
```

---

## 🧠 Model Architecture

```
Input (224 × 224 × 3)
    │
    ▼
ResNet50 Backbone (ImageNet V2 pre-trained)
    │
    ▼
Identity() — removes original FC layer
    │
    ▼
Linear(2048 → 256) → BatchNorm → ReLU → Dropout(0.4)
    │
    ▼
Linear(256 → 128) → BatchNorm → ReLU → Dropout(0.4)
    │
    ▼
Linear(128 → 5) — Output (5 DR severity classes)
```

---

## 📈 Training Strategy

### Two-Phase Transfer Learning

| Phase | Epochs | Strategy               | Learning Rate | Description                    |
|-------|--------|------------------------|---------------|--------------------------------|
| **1** | 1–10   | Frozen Backbone        | 1e-4          | Only classifier head trains    |
| **2** | 11–20  | Fine-tuning            | 1e-5          | Backbone layers 3 & 4 unfrozen |

- **Optimizer**: Adam (weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: CrossEntropyLoss
- **GPU**: NVIDIA RTX 3050 Laptop (4.3 GB VRAM)

---

## 📊 Results

| Metric                    | Value    |
|---------------------------|----------|
| **Overall Test Accuracy** | 77.57%   |
| **F1 Score (Weighted)**   | 72.90%   |
| **Best Val Accuracy**     | 76.90%   |
| **Total Epochs**          | 20       |

### Per-Class Performance

| Class         | Precision | Recall | F1-Score | Support |
|---------------|-----------|--------|----------|---------|
| Mild          | 0.5466    | 0.3249 | 0.4076   | 794     |
| Moderate      | 0.8137    | 0.9636 | 0.8823   | 3,872   |
| Severe        | 0.4470    | 0.5566 | 0.4958   | 106     |
| Proliferative | 0.4875    | 0.2977 | 0.3697   | 131     |

> **Note**: The dataset is heavily imbalanced — Moderate class has ~73% of all images, which inflates overall accuracy but affects minority class performance.

---

## 🌐 Web Application

The project includes a **Flask web app** with a modern dark-themed UI:

- **Drag & drop** or click to upload retinal fundus images
- **Real-time prediction** with confidence scores
- **Severity color coding** (green → red based on severity)
- **Probability bars** for all 5 classes
- **Clinical description** for each severity level

```bash
python app.py
# Visit http://127.0.0.1:5000
```

---

## 🛠 Technologies

| Tool              | Purpose                              |
|-------------------|--------------------------------------|
| **PyTorch 2.5**   | Deep learning framework              |
| **torchvision**   | Pre-trained models & transforms      |
| **scikit-learn**   | Metrics & evaluation                |
| **Flask**         | Web application framework            |
| **Matplotlib**    | Visualization                        |
| **Seaborn**       | Statistical plots                    |
| **Pillow**        | Image processing                     |
| **NumPy / Pandas**| Data handling                        |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not** a medical diagnostic tool. Always consult a qualified ophthalmologist for clinical decisions.

---

## 📄 License

MIT License — feel free to use and modify for learning and research.
