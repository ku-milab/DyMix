# DyMix

This repository provides the official implementation of DyMix, a dynamic frequency Mixup scheduling strategy for unsupervised domain adaptation (UDA) in medical image classification, as described in the following paper:
> **DyMix: Dynamic Frequency Mixup Scheduler based Unsupervised Domain Adaptation for Enhancing Alzheimer's Disease Prediction**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1</sup>, [Yooseung Shin](https://scholar.google.co.kr/citations?user=yCvN9Z8AAAAJ&hl=ko)<sup>1, 2</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Heuron Company Ltd.) <br/>
> 
> **Abstract:** *Recent advances in deep learning (DL) have substantially improved the accuracy of Alzheimer’s disease (AD) diagnosis from brain images, enabling earlier and more reliable clinical interventions. Nevertheless, most DL-based models often suffer from significant performance degradation when applied to unseen domains owing to variations in data distributions, a challenge commonly referred to as domain shift. To address this issue, we propose DyMix, a dynamic frequency Mixup scheduler for unsupervised domain adaptation (UDA). Built upon a Fourier transformation, DyMix dynamically adjusts the frequency components between source and target domains within selected regions, allowing the model to efficiently capture domain-relevant information. To further enhance robustness, DyMix incorporates intensity-invariant learning and self-adversarial regularization, encouraging the extraction of stable and domain-invariant feature representations. Such an adaptive framework enables robust cross-domain generalization by dynamically aligning domain-specific frequency characteristics while maintaining informative disease-relevant representations. Extensive experiments on two benchmark datasets (i.e., ADNI and AIBL) demonstrate that DyMix consistently outperforms state-of-the-art UDA methods for AD diagnosis. As a result, our method has achieved average performance gains of +6.04% in accuracy and +5.88% in AUC compared to the mean score of all baseline methods across multiple cross-domain scenarios.*
>
> 
## Overview
DyMix is designed to address domain shift in multi-site MRI data by dynamically adjusting the frequency mixing region during training.
The framework consists of two stages:

1. **Pretraining stage**
  - Coarse intensity augmentation via amplitude–phase recombination (APR)
  - Self-adversarial learning for intensity-invariant feature extraction

2. **Adaptation stage**
  - Dynamic frequency Mixup guided by a validation-based scheduler
  - Fine-grained frequency alignment between source and target domains

## Data Preprocessing Pipeline
This repository provides the complete preprocessing pipeline used in our experiments on the **ADNI** and **AIBL** datasets.
Due to data usage and licensing restrictions, **raw or preprocessed MRI data are not redistributed**.
Instead, we release all preprocessing scripts and detailed instructions to ensure full reproducibility for researchers with authorized access to the datasets.

### Overview
All brain MRI scans from ADNI and AIBL were identically preprocessed using a unified pipeline to minimize domain-specific biases and ensure consistent input representations across datasets.

### 1. Brain extraction

Non-brain tissues (e.g., skull, neck, and surrounding structures) are removed using **HD-BET**:

- Tool: HD-BET
- Purpose: Robust skull stripping with minimal manual intervention

```bash
hd-bet -i input.nii.gz -o brain.nii.gz -device cpu
```

### 2. Linear Registration to MNI152 Space
The skull-stripped images are aligned to the **MNI152 standard template** using **FLIRT** from FSL v6.0.1.

- Registration type: Linear (affine)
- Degrees of freedom: 12 (translation, rotation, scale, shear)
- Reference space: MNI152

```bash
flirt -in brain.nii.gz \
      -ref MNI152_T1_1mm.nii.gz \
      -out brain_mni.nii.gz \
      -omat affine.mat
```
This step corrects global linear differences across subjects and datasets.

### 3. Spatial Normalization and Resampling
All registered volumes are resampled to a **uniform spatial resolution** of 1 x 1 x 1 mm³

### 4. Intensity Normalization

To reduce intensity scale variations across scanners and acquisition protocols, **Min–Max normalization** is applied voxel-wise.

### Output Format
Each preprocessed sample is stored as a 3D NIfTI volume with the following properties:

- Shape: `193 x 229 x 193`
- Resolution: `1 mm³ isotropic`
- Intensity range: `[0, 1]`
- Coordinate space: `MNI152`

## Data Licensing and Availability
> **ADNI** (https://adni.loni.usc.edu): Data usage is governed by the ADNI Data Use Agreement
> 
> **AIBL** (https://aibl.csiro.au): Data usage follows AIBL access and licensing policies
