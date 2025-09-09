# Selection-Based Histopathology Cancer Detection

## Introduction
Breast cancer is one of the most prevalent and deadly cancers worldwide. Histopathology images play a crucial role in early detection and diagnosis, yet manual inspection is often subjective and time-consuming.

This project presents a deep learning + feature selection framework for automated breast cancer detection from histopathology images. The framework integrates CNN-based feature extraction with selection strategies to improve classification performance and computational efficiency.

Key contributions:

- Fine-tuned CNN pretrained on CIFAR-100 for histopathology image feature extraction.
- Feature selection pipeline combining R-Relief and Pearson Correlation Coefficient (PCC), followed by PCA for dimensionality reduction.
- Comprehensive evaluation with multiple classifiers: SVMs (various kernels), Decision Trees, and Ensembles.

---

## Background

- CNNs: Extract rich, high-level features from medical images.
- Raw features: Raw deep features are high-dimensional and redundant.
- Solution: Feature selection to improve accuracy and efficiency.

Methods used:
    - R-Relief: Ranks discriminative features.
    - PCC: Removes redundant/correlated features.
    - PCA: Fuses selected features into a compact representation.

---

## Methodology
### Step 1: CNN Training
- Pretrained CNN model on CIFAR-100 (60,000 images, 100 classes).
- Fine-tuned on histopathology dataset.
- Training details:
    
    - Epochs: 60
    - Batch size: 128
    - Initial learning rate: 0.01
    - Optimizer: SGD with momentum

### Step 2: Feature Extraction
- Extract deep feature vectors (DF) from the last fully connected layers.
- Each histopathology image to get feature vectors.

### Step 3: Feature Selection
- **Parallel selection strategy**:

    - Apply R-Relief to rank features by discriminative power.
    - Apply PCC to remove redundant / irrelevant features.

- **Fusion step**: PCA to generate final compact feature vector (FV).

### Step 4: Classification
- Multiple classifiers trained on selected features:

    - Linear SVM (LSVM)
    - Quadratic SVM (QSVM)
    - Cubic SVM (CSVM)
    - Gaussian SVM (Fine, Medium, Coarse)
    - Fine Tree (FT)
    - Ensemble Boosted Trees (EBT)
    - Ensemble Subspace Discriminant (ESD)

- Evaluation metric: Accuracy, Precision, Recall, F1-score (5-fold cross-validation).

---

## Dataset
- Dataset used: IDC Breast Cancer Histopathology dataset.
- Classes: IDC Positive vs IDC Negative.
- Preprocessing:

    - Resize images to fixed dimensions.
    - Normalization to [0,1].
    - Train/test split with stratification.

## Quick Start

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Run the complete pipeline
```
jupyter notebook cancer_selection_pipeline.ipynb
```

## Experiments
- Baseline: CNN features without FS.
- Comparison: CNN features with FS (R-Relief + PCC + PCA).
- Evaluation: 5-fold cross-validation.

## Results

## Limitations & Future Work

### 1. Limitations
- Classification relied only on deep features, without combining them with traditional features.
- No data augmentation methods were applied.
- No preprocessing steps were introduced to improve input quality.

### 2. Future work:
- Apply preprocessing techniques and feature fusion across multiple domains to enhance accuracy.
- Develop an intelligent CAD system for IDC classification.
- Explore advanced CNN models such as DenseNet and CapsuleNet, integrated with diverse feature fusion and selection strategies.

## Citation
If you use this repository, please cite:

```bibtex
@article{selectioncancer2022,
  title={A Framework of Deep Learning and Selection-Based Breast Cancer Detection from Histopathology Images},
  author={Muhammad Junaid Umer, Muhammad Sharif, Majed Alhaisoni, Usman Tariq, Ye Jin Kim and Byoungchol Chang},
  journal={Computer Systems Science & Engineering},
  year={2022},
  publisher = {Computer Systems Science & Engineering}
}
```