# A versatile deep-learning based strategy for multi-omics integration

## Overview

CustOmics is a Python package for integrating multiple genomic data modalities (e.g., RNA-seq, CNV, and DNA methylation) using a hierarchical deep-learning architecture. It supports classification, survival outcome prediction, and SHAP-based explainability, all in a single scikit-learn-style API.

CustOmics is designed to provide a modern and research-friendly framework for computational biology and precision medicine.

## Architecture

``` mermaid
flowchart LR
    %% Input omics
    RNA["RNA-Seq"] --> AE1
    CNV["Copy Number Variations (CNV)"] --> AE2
    METH["DNA Methylation"] --> AE3

    %% Source-specific autoencoders
    subgraph Phase1["Phase 1 — Source-specific Representation Learning"]
        direction TB
        AE1["RNA-Seq Autoencoder"]
        AE2["CNV Autoencoder"]
        AE3["Methylation Autoencoder"]
    end

    %% Central integration
    AE1 --> CVAE
    AE2 --> CVAE
    AE3 --> CVAE

    subgraph Phase2["Phase 2 — Mixed Integration"]
        CVAE["Central Variational Autoencoder\nShared Latent Representation"]
    end

    %% Downstream tasks
    CVAE --> CLS["Tumor / Subtype Classification"]
    CVAE --> SURV["DeepSurv Survival Prediction"]

    %% Styling
    classDef omics fill:#E8F0FE,stroke:#4A90E2,stroke-width:2,color:#000;
    classDef ae fill:#EAF7EA,stroke:#34A853,stroke-width:2,color:#000;
    classDef latent fill:#FFF4D6,stroke:#FBBC05,stroke-width:3,color:#000;
    classDef task fill:#FCE8E6,stroke:#EA4335,stroke-width:2,color:#000;

    class RNA,CNV,METH omics;
    class AE1,AE2,AE3 ae;
    class CVAE latent;
    class CLS,SURV task;

```
At the core of Costomics is a two-phase mixed-integration workflow:

- **Phase 1** trains per-source autoencoders jointly with the task heads.
- **Phase 2** additionally trains the central VAE to consolidate the integrated representation.

## Why use `CustOmics`

`CustOmics` is built for high-dimensional and heterogeneous omics datasets.

### 1. Unified API

- Classification tasks
- Survival outcome prediction
- Latent representation learning
- SHAP-based explainability
- Modular source-specific autoencoders for flexible experimentation
- Scalable training using PyTorch and GPU acceleration
- Compatible with scikit-learn-style workflows

### 2. Visualization Utilities

- Latent space exploration
- Kaplan-Meier survival stratification
- Feature attribution analysis
- Easily extensible to custom architectures, tasks, and omics sources

Start using `CustOmics` by reading the [getting started](getting_started) guide.
