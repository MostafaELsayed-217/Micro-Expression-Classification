# 🧠 Micro-Expression Classification using Deep Learning

> Facial micro-expression recognition using CNNs, ResNet, and VGG16 — achieving **76.9% accuracy** on a 7-class emotion classification task, **4.9% above the 72% baseline**.

---

## 📌 Overview

Micro-expressions are involuntary, short-duration facial expressions (under 500ms) that reveal a person's true emotional state even when they attempt to conceal it. This project builds an end-to-end deep learning pipeline to automatically classify micro-expressions into **7 emotion classes**.

Beyond model development, this project contributes a **new micro-expression dataset** to the research community — addressing one of the most critical bottlenecks in the field: data scarcity.

**Real-world applications:**
- 🏥 Mental health tech & clinical assessment
- 🧑‍💼 HR screening & behavioral analysis
- 🤖 Human-Computer Interaction (HCI)
- 🔍 Deception detection & security

---

## 📊 Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| VGG16 | 64% | Epoch = 30 |
| AlexNet | 65% | Epoch = 30 |
| VGG16 + Optical Flow | — | State-of-the-art comparison |
| **Our Best Model (CNN + ResNet)** | **76.9%** | **4.9% above 72% baseline** |

---

## 🏗️ Architecture & Pipeline

```
Raw Video Data
      │
      ▼
Face Detection & Alignment (Viola-Jones)
      │
      ▼
Preprocessing
  ├── Motion Magnification (enhance subtle movements)
  ├── Optical Flow Extraction (temporal information)
  ├── Region of Interest (ROI) Extraction
  └── Data Augmentation (flipping, rotation, temporal jittering)
      │
      ▼
Feature Extraction
  ├── CNN (spatial features)
  ├── ResNet (deep residual learning)
  └── VGG16 (transfer learning from ImageNet)
      │
      ▼
Temporal-Attention Module
(focus on key frame changes in temporal space)
      │
      ▼
SoftMax Classifier → 7 Emotion Classes
```

**Emotion Classes:** Happiness · Sadness · Surprise · Fear · Disgust · Anger · Contempt

---

## 🗂️ Dataset

This project uses **CASME II** (one of the most widely used spontaneous micro-expression datasets) as the primary training dataset, alongside a **new dataset collected and contributed by our team**.

### Our Dataset Contribution
- Collected in a controlled lab environment (university campus)
- Elicited using video stimuli (standard methodology)
- Labelled using the **Facial Action Coding System (FACS)**
- Structured and ready for publishing to the research community

### Why a New Dataset?
Existing micro-expression datasets suffer from:
- Limited number of subjects
- Class imbalance
- Inconsistency in expression elicitation methods
- Low frame rate (micro-expressions require high-speed cameras)

---

## ⚙️ Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Architectures | CNN, ResNet, VGG16, AlexNet |
| Feature Extraction | Optical Flow, Apex Frame |
| Preprocessing | Motion Magnification, FACS-based ROI |
| Evaluation | Accuracy, F1-Score, UAR, UF1 |
| Validation Strategy | SD-LOSO (Subject-Dependent Leave-One-Subject-Out) |

---

## 📁 Project Structure

```
micro-expression-classification/
│
├── data/
│   ├── raw/                  # Original video sequences
│   ├── processed/            # Preprocessed frames
│   └── optical_flow/         # Extracted optical flow maps
│
├── models/
│   ├── cnn_baseline.py       # CNN baseline model
│   ├── resnet_model.py       # ResNet-based model
│   └── vgg16_model.py        # VGG16 transfer learning
│
├── preprocessing/
│   ├── face_detection.py     # Viola-Jones face detection
│   ├── optical_flow.py       # Optical flow extraction
│   ├── augmentation.py       # Data augmentation pipeline
│   └── roi_extraction.py     # Region of interest extraction
│
├── evaluation/
│   └── metrics.py            # Accuracy, F1, UAR, UF1
│
├── results/                  # Training logs and plots
└── README.md
```

---

## 📈 Training Details

- **Dataset split:** SD-LOSO cross-validation (leave-one-subject-out)
- **Augmentation:** Flipping, rotation, temporal jittering (100%, 90%, 80%, 70%, 60%, 50% frame sampling)
- **Transfer Learning:** VGG16 and ResNet pre-trained on ImageNet, fine-tuned on micro-expression data
- **Optimizer:** Adam
- **Loss:** Categorical Cross-Entropy
- **Evaluation Metrics:** Accuracy, Weighted F1, UAR (Unweighted Average Recall), UF1

---

## 🔬 Research Context

This project is part of a graduation research effort at **Alexandria University, Faculty of Computing and Data Sciences**, supervised by **Prof. Dr. Khaled Mahar**.

The work involved an extensive literature review covering:
- Spatial methods (CNN, ResNet, VGG16, AlexNet)
- Temporal methods (RNN, RCNN, STRCN)
- Spatial-temporal methods (3D-CNN, STSTNet, ATNet)
- Multi-stream networks (OFF-ApexNet, LEARNet, TSCNN)

---

## 👨‍💻 Author

**Mostafa El-Sayed** — Computer Vision Engineer & Data Scientist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/MostafaELsayed-217)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/MostafaELsayed-217)

---

## 📄 Citation

If you use this work or dataset in your research, please cite:

```
El-Sayed, M. et al. (2023). Micro-Expression Classification with Deep Learning.
BSc. Final Project Report, Alexandria University, Faculty of Computing and Data Sciences.
Supervised by Prof. Dr. Khaled Mahar.
```
