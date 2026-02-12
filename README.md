# Transformers-for-Thoracic-Diagnosis

**Benchmarking ViT, Swin & DeiT on CheXpert**

## Overview

This project investigates the effectiveness of **Transformer-based architectures** for automated thoracic disease classification from chest X-rays (CXRs). While CNNs dominate medical imaging, Transformers offer improved global context modeling through self-attention mechanisms.

We benchmark **Vision Transformers (ViT), Swin Transformers, DeiT, and a hybrid ResNet+ViT** on the **CheXpert dataset (224K+ images, 14 labels)** to evaluate performance, stability, and fine-tuning strategies.

---

## Dataset

* **Dataset:** CheXpert (Stanford ML Group)
* **Samples:** 224,000+ chest radiographs
* **Labels:** 14 thoracic diseases (multi-label classification)
* **Class Imbalance:** Severe positive/negative skew handled via `WeightedRandomSampler`

---

## Models Evaluated

* ViT (Base & Large)
* Swin Transformer
* Data-efficient Image Transformer (DeiT)
* ResNet + ViT Hybrid

---

## Fine-Tuning Strategies

1. Classification head only
2. Head + last Transformer block
3. Full model fine-tuning

---

## Training Details

* **Optimizer:** AdamW
* **Weight Decay:** 0.01
* **Hardware:** NVIDIA H100 GPUs
* **Task:** Multi-label classification
* **Evaluation Metric:** Validation AUC

---

## Results

* Shallow tuning (classification head only) produced the most stable performance.
* Achieved **Validation AUC between 0.65 – 0.67**.
* Full fine-tuning showed higher overfitting and instability compared to CNN baselines.

---

## Model Interpretability

We applied **Grad-CAM** to visualize attention regions and ensure models focus on clinically relevant anatomical structures.

---

## Key Takeaways

* Transformer models provide high capacity but are sensitive to overfitting in medical imaging.
* Fine-tuning strategy significantly impacts stability.
* CNNs remain competitive in low-data or highly imbalanced clinical settings.

---

# Vision Transformer Fine-Tuning Scripts

This repository contains scripts for fine-tuning Vision Transformer (ViT) models with different strategies.

## Files and Their Purpose

### 1. `ViT_Classification_head_Final_TF_Block_trainable.py`

* **Trainable Parts:** Classification head + final Transformer block
* **Description:** Fine-tunes the last Transformer block along with the classification head. This allows the model to adjust high-level features while training the output layer for your specific task.
* **Use Case:** Medium dataset sizes where some feature adaptation is needed.
* **Risk:** Moderate overfitting risk compared to training only the head.

---

### 2. `ViT_Classification_head_trainable.py`

* **Trainable Parts:** Only the classification head
* **Description:** Keeps the ViT backbone frozen and trains only the output layer.
* **Use Case:** Small datasets where stable and reliable training is required.
* **Risk:** Low overfitting risk.

---

### 3. `ViT_unfreeze_all.py`

* **Trainable Parts:** Entire ViT model
* **Description:** Fully fine-tunes the model, allowing all layers to adapt to the dataset.
* **Use Case:** Large datasets or tasks requiring full adaptation.
* **Risk:** High risk of overfitting and unstable training on smaller datasets.

---
