# MRI-Based Detection of Alzheimer's Disease Using Transfer Learning in CNNs

> Capstone Project — Department of Computer Science, Ashoka University  
> **Author:** Aarushi Basu  
> **Supervisors:** Prof. Anirban Sen & Prof. Dipanjan Ray  
> **Date:** December 8, 2024

---

## Overview

This project presents a CNN-based architecture for classifying Alzheimer's disease (AD) from brain MRI scans using transfer learning. The core motivation is the limited availability of Alzheimer's imaging data — transfer learning from pathologically similar diseases (dementia, MCI) is used to improve classification performance even on small datasets.

The model achieves **95% accuracy** in detecting AD from normal control, outperforming clinical visual rating scales (~82% diagnostic performance).

---

## Table of Contents

- [Background](#background)
- [Proposed Contributions](#proposed-contributions)
- [Datasets](#datasets)
- [Models](#models)
- [Results](#results)
- [Challenges & Mitigations](#challenges--mitigations)
- [Future Work](#future-work)
- [References](#references)

---

## Background

### Alzheimer's Disease
Alzheimer's disease affects ~55 million people worldwide and accounts for over 60% of dementia cases. This number is projected to reach 139 million by 2050. The disease is characterised by progressive brain atrophy — particularly in the hippocampus, amygdala, entorhinal cortex, and parahippocampal cortices — caused by beta-amyloid plaques and neurofibrillary tau tangles.

Early diagnosis (preclinical stage) is critical because there is currently no treatment once the disease progresses past this stage. However, early-stage symptoms are subtle and easily confused with related conditions like MCI and dementia — making deep learning a compelling screening tool.

### Why CNNs and Transfer Learning?
CNNs excel at image classification by learning hierarchical spatial features — from edges and textures in early layers to complex structural patterns in deeper layers. Transfer learning allows a model pre-trained on a related task to adapt to a new one with far less data, saving computational resources and reducing overfitting risk.

---

## Proposed Contributions

1. **CNN architecture optimisation** — Identified the minimal viable architecture for image classification via experiments on CIFAR-10 (25+ configurations tested).
2. **Hyperparameter tuning** — Determined optimal optimizer (Adam), loss function (CrossEntropyLoss), learning rate (0.001), batch size, and epoch count.
3. **Transfer learning benchmarking** — Compared standard CNN with pretrained VGGNet, ResNet50, and EfficientNet on CIFAR-10.
4. **Pathological correlation** — Established structural MRI similarity between Dementia/MCI and Alzheimer's disease (Models 2 & 4), motivating cross-disease transfer learning.
5. **AD detection via transfer learning** — Final models (3 & 5) leverage pretraining on dementia/MCI data to classify AD from normal control with high accuracy.

---

## Datasets

### ADNI (Alzheimer's Disease Neuroimaging Initiative)
A multi-site US/Canada study tracking Alzheimer's and MCI across multiple modalities. The version used here is a preprocessed Kaggle release of the ADNI dataset.

| Class | Description |
|-------|-------------|
| AD | Alzheimer's Disease (44.5%) |
| MCI | Mild Cognitive Impairment (6.2%) |
| EMCI | Early MCI (2.6%) |
| LMCI | Late MCI |
| Control | Normal Control (46.1%) |

Data format: volumetric T1-weighted MR images recorded at 1.5T.

### Dementia Dataset
An Alzheimer's disease multiclass dataset tracking disease progression from early dementia signs, sourced from Kaggle.

| Class | Distribution |
|-------|-------------|
| Mild Dementia | 22.7% |
| Moderate Dementia | 22.7% |
| Very Mild Dementia | 25.5% |
| Non-Demented | 29.1% |

---

## Preprocessing Pipeline

All models apply the following preprocessing steps:

1. **Z-score normalisation** — Standardises image brightness and contrast across scans.
2. **Data augmentation** — Handles slight rotations, flips, and distortions to improve generalisation.
3. **Skull stripping** — Removes non-brain tissue so the CNN focuses exclusively on brain structures.

---

## Models

All models use the **Adam optimizer**, **learning rate 0.001**, **batch size 32**, **50 epochs**, and **10-fold cross-validation** unless noted otherwise.

---

### Model 1 — Binary AD vs. Normal Control (Baseline CNN)

**Task:** Binary classification — Alzheimer's vs. Normal Control  
**Architecture:** 13 layers — Conv(32) → BN → Pool → Conv(64) → BN → Pool → Flatten → Dense(512) → Dropout(0.25) → Dense(128) → Dropout(0.25) → Softmax output  
**Dataset:** 500 AD + 500 Control (train), 200 AD + 200 Control (test)  
**Loss:** Categorical cross-entropy

---

### Model 2 — Dementia-Trained, AD-Tested

**Task:** Establish structural similarity between dementia and AD pathology  
**Architecture:** 10 layers — Conv(32) → Pool → Conv(64) → Pool → Flatten → Dense(128) → Dropout(0.25) → Sigmoid output  
**Dataset:** Trained on 1780 Dementia + 1780 Control; tested on 200 AD + 200 Control  
**Loss:** Binary cross-entropy

---

### Model 3 — Transfer Learning: Dementia → AD (Best Dementia-Based Model)

**Task:** Binary AD classification using features transferred from a dementia 5-class model  
**Architecture (5-class):** 3 Conv blocks [Conv → Pool → Dropout(0.25)] → Flatten → Dense(256) → Dropout → Softmax(5)  
**Architecture (binary):** Frozen 5-class feature extractor → Dense(128) → Dropout(0.25) → Sigmoid  
**Dataset (5-class):** Mild/Moderate/Very Mild Dementia + Control + AD (70/15/15 split)  
**Dataset (binary):** 500 AD + 4060 non-AD (train); 200 AD + 200 Control (test)

---

### Model 4 — MCI-Trained, AD-Tested

**Task:** Establish structural similarity between MCI and AD  
**Architecture:** Same as Model 2  
**Dataset:** Trained on 1780 MCI + 1780 Control; tested on 200 AD + 200 Control  
**Loss:** Binary cross-entropy

---

### Model 5 — Transfer Learning: MCI → AD (Best Overall Model)

**Task:** Binary AD classification using features transferred from a MCI 5-class model  
**Architecture:** Same structure as Model 3, but pretrained on MCI/EMCI/LMCI/Control/AD data  
**Dataset (5-class):** EMCI(336) + LMCI(100) + MCI(808) + Control(1596) + AD(350) for training  
**Dataset (binary):** 500 AD + 4060 non-AD (train); 200 AD + 200 Control (test)

---

## Results

| Model | Approach | Accuracy | Key Finding |
|-------|----------|----------|-------------|
| Model 1 | Baseline CNN (AD vs. Control) | 80% | Solid baseline; 22% of AD cases missed |
| Model 2 | Dementia-trained, AD-tested | ~55% | High AD misclassification as dementia — confirms pathological overlap |
| Model 3 | Transfer learning from dementia | **90%** | Transfer learning significantly improves recall for AD |
| Model 4 | MCI-trained, AD-tested | ~75% | MCI–AD similarity confirmed; recall for AD very low (0.14) |
| Model 5 | Transfer learning from MCI | **95%** | Best model — recall 0.98, F1 0.97 |

### Model 5 — Detailed Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 95% |
| Precision (non-AD) | 85% |
| Recall (AD) | 0.98 |
| F1 Score | 0.97 |

Model 5 correctly identified 98% of true AD cases, making it highly effective at minimising missed diagnoses. Training and validation loss curves converge cleanly, indicating no significant overfitting.

---

## Challenges & Mitigations

### Challenges
- **Limited data** — Very few publicly available Alzheimer's datasets; existing ones are small and not easily generalisable.
- **Motion artefacts** — Patient movement during scanning causes blurring and false structural anomalies.
- **Physiological variation** — Individual brain differences make separating pathological patterns difficult.
- **Non-standardised equipment** — Scanners across hospitals lack consistent protocols.
- **Overlapping pathology** — AD shares structural features with dementia, MCI, and stroke, making subtle differences hard to learn.
- **Class imbalance** — Existing datasets are heavily imbalanced, particularly in preclinical stages.

### Mitigations
- Datasets were **manually balanced** across classes to prevent class imbalance effects.
- **Transfer learning** from dementia/MCI data compensates for limited AD samples.
- **10-fold cross-validation** ensures robust, generalisable performance estimates.
- Preprocessing pipeline (Z-normalisation, augmentation, skull stripping) reduces noise and improves consistency.

---

## Future Work

- **Grad-CAM visualisation** — Highlight which brain regions drive classification decisions, improving clinical interpretability.
- **Larger and multi-modal datasets** — Incorporate PET, genetic, and CSF biomarker data alongside MRI.
- **3D CNN extensions** — Adapt 2D filters to 3D to better capture volumetric brain changes.
- **Fine-grained staging** — Extend the model to classify preclinical AD stages more precisely.

---

## References

Key references include:

- Alzheimer's Disease International. (2020). *Dementia Statistics.*
- Ebrahimi et al. (2021). Deep sequence modelling for Alzheimer's disease detection using MRI. *Computers in Biology and Medicine.*
- Liu et al. (2020). A multi-model deep CNN for automatic hippocampus segmentation in Alzheimer's disease. *NeuroImage.*
- Mehmood et al. (2021). A Transfer Learning Approach for Early Diagnosis of Alzheimer's Disease on MRI Images. *Neuroscience.*
- Murugan et al. (2021). DEMNET: A Deep Learning Model for Early Diagnosis of Alzheimer Diseases. *IEEE Access.*
- Yamashita et al. (2018). Convolutional Neural Networks: an Overview and Application in Radiology. *Insights into Imaging.*

Full reference list available in the project report.

---

## Acknowledgements

Special thanks to **Prof. Anirban Sen** and **Prof. Dipanjan Ray** for their guidance throughout this project, and to family and friends for their continued support.
