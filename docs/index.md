# A versatile deep-learning based strategy for multi-omics integration

## Overview

CustOmics is a Python package for integrating multiple genomic data modalities (RNA-seq, CNV, DNA methylation, …) using a hierarchical deep-learning architecture. It supports classification, survival outcome prediction, and SHAP-based explainability — all in a single scikit-learn-style API.

Costomics is designed to provide a modern and research-friendly framework for computational biology and precision medicine.

## Architecture

``` mermaid
graph LR
  A[RNA-seq] --> B{AE_1} --> G{Central VAE - latent space} --> I[Classifier];
  C[CNV] --> E{AE_2} --> G{Central VAE - latent space} --> H[Survival predictor];
  D[Methyl] --> F{AE_3} --> G{Central VAE - latent space};

```
At the core of Costomics is a two-phase mixed-integration workflow:

- **Phase 1** trains per-source autoencoders jointly with the task heads.
- **Phase 2** additionally trains the central VAE to consolidate the integrated representation.

## Why use `CustOmics`

`CustOmics` is built for high-dimensional and heterogeneous omics datasets.

> #### Unified API for:
>    - Classification tasks
>    - Survival outcome prediction
>    - Latent representation learning
>    - SHAP-based explainability
>    - Modular source-specific autoencoders for flexible experimentation
>    - Scalable training using PyTorch and GPU acceleration
>    - Compatible with scikit-learn-style workflows


> #### Includes visualization utilities for:
>   - Latent space exploration
>   - Kaplan–Meier survival stratification
>   - Feature attribution analysis
>   - Easily extensible to custom architectures, tasks, and omics sources

Start using Sopa by reading our [getting started](getting_started) guide!
