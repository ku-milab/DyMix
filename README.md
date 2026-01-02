# DyMix

This repository provides a PyTorch implementation of the following paper:
> **DyMix: Dynamic Frequency Mixup Scheduler based Unsupervised Domain Adaptation for Enhancing Alzheimer's Disease Prediction**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1</sup>, [Yooseung Shin](https://scholar.google.co.kr/citations?user=yCvN9Z8AAAAJ&hl=ko)<sup>1, 2</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Heuron Company Ltd.) <br/>
> 
> **Abstract:** *Recent advances in deep learning (DL) have substantially improved the accuracy of Alzheimer’s disease (AD) diagnosis from brain images, enabling earlier and more reliable clinical interventions. Nevertheless, most DL-based models often suffer from significant performance degradation when applied to unseen domains owing to variations in data distributions, a challenge commonly referred to as domain shift. To address this issue, we propose DyMix, a dynamic frequency Mixup scheduler for unsupervised domain adaptation (UDA). Built upon a Fourier transformation, DyMix dynamically adjusts the frequency components between source and target domains within selected regions, allowing the model to efficiently capture domain-relevant information. To further enhance robustness, DyMix incorporates intensity-invariant learning and self-adversarial regularization, encouraging the extraction of stable and domain-invariant feature representations. Such an adaptive framework enables robust cross-domain generalization by dynamically aligning domain-specific frequency characteristics while maintaining informative disease-relevant representations. Extensive experiments on two benchmark datasets (i.e., ADNI and AIBL) demonstrate that DyMix consistently outperforms state-of-the-art UDA methods for AD diagnosis. As a result, our method has achieved average performance gains of +6.04% in accuracy and +5.88% in AUC compared to the mean score of all baseline methods across multiple cross-domain scenarios.*
>
> 
