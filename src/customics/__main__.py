"""CLI entry point: ``python -m customics`` or ``customics`` command."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="customics",
        description="Train and evaluate a CustOMICS multi-omics integration model.",
    )
    parser.add_argument("-c", "--cohorts", type=str, required=True,
                        help="Cohort name(s), e.g. TCGA-BRCA")
    parser.add_argument("-dv", "--device", type=str, default="cpu",
                        help="Compute device: 'cpu' or 'cuda'")
    parser.add_argument("-dr", "--data_directory", type=str, default="data/",
                        help="Root directory containing omics and clinical data")
    parser.add_argument("-res", "--result_directory", type=str, default="results/",
                        help="Directory for saving results and plots")
    parser.add_argument("-t", "--task", type=str,
                        choices=["classification", "survival"], default="classification")
    parser.add_argument("-src", "--sources", type=str, default="CNV,RNAseq,methyl",
                        help="Comma-separated omics sources to integrate")
    parser.add_argument("-nc", "--num_classes", type=int, default=4)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-p2", "--p2_switch", type=int, default=10)
    parser.add_argument("-lr", "--lr", type=float, default=1e-3)
    parser.add_argument("-bt", "--beta", type=float, default=1.0)
    parser.add_argument("-dp", "--dropout", type=float, default=0.2)
    parser.add_argument("-hd", "--hidden_dim", type=str, default="512,256")
    parser.add_argument("-ct", "--central_dim", type=str, default="512,256")
    parser.add_argument("-lt", "--latent_dim", type=int, default=128)
    parser.add_argument("-ch", "--classifier_dim", type=str, default="128,64")
    parser.add_argument("-sh", "--survival_dim", type=str, default="64,32")
    parser.add_argument("-lc", "--lambda_classif", type=float, default=5.0)
    parser.add_argument("-ls", "--lambda_survival", type=float, default=1.0)

    args = parser.parse_args()

    if args.device != "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    sources = args.sources.split(",")
    hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    central_dim = [int(x) for x in args.central_dim.split(",")]
    classifier_dim = [int(x) for x in args.classifier_dim.split(",")]
    survival_dim = [int(x) for x in args.survival_dim.split(",")]

    print(
        f"customics v0.1.0 — task={args.task}, cohort={args.cohorts}, "
        f"sources={sources}, device={device}"
    )
    print("Please use the Python API or the example notebook for full pipeline usage.")
    sys.exit(0)


if __name__ == "__main__":
    main()
