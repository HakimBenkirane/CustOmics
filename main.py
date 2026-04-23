"""Legacy CLI entry point — use the ``customics`` command or Python API instead.

This script is kept for backwards compatibility.  For new usage, prefer:

    customics --help

or the Python API documented in README.md.
"""

import argparse
import sys

import numpy as np
import torch

from customics import CustOMICS
from customics.tools.utils import get_common_samples, get_sub_omics_df, save_splits, get_splits


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a customics multi-omics model."
    )
    parser.add_argument("-c", "--cohorts", type=str, required=True)
    parser.add_argument("-dv", "--device", type=str, default="cpu")
    parser.add_argument("-dr", "--data_directory", type=str, default="data/")
    parser.add_argument("-res", "--result_directory", type=str, default="results/")
    parser.add_argument("-t", "--task", type=str,
                        choices=["classification", "survival"], default="classification")
    parser.add_argument("-src", "--sources", type=str, default="CNV,RNAseq,methyl")
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
    return parser.parse_args()


def train(args, device, sources, hidden_dim, central_dim, classifier_dim, survival_dim):
    """Run 5-fold cross-validation for the given cohort and task."""
    import pandas as pd
    import os

    data_dir = args.data_directory
    label_col = "PAM50" if args.cohorts == "TCGA-BRCA" else "tumor_type"
    event_col = "OS"
    surv_time_col = "OS.time"

    # Load omics data — adjust paths to match your local directory layout
    omics_df = {}
    for source in sources:
        path = os.path.join(data_dir, args.cohorts, f"{source}.tsv")
        if os.path.exists(path):
            omics_df[source] = pd.read_csv(path, sep="\t", index_col=0)
        else:
            raise FileNotFoundError(
                f"Expected omics file not found: {path}\n"
                "Please organise your data as data/<cohort>/<source>.tsv"
            )

    clinical_path = os.path.join(data_dir, args.cohorts, "clinical.tsv")
    clinical_df = pd.read_csv(clinical_path, sep="\t", index_col=0)

    lt_samples = get_common_samples(list(omics_df.values()) + [clinical_df])
    save_splits(lt_samples, args.cohorts)

    x_dim = [omics_df[s].shape[1] for s in sources]
    num_classes = clinical_df[label_col].nunique()

    source_params = {
        s: {"input_dim": d, "hidden_dim": hidden_dim, "latent_dim": args.latent_dim,
            "norm": True, "dropout": args.dropout}
        for s, d in zip(sources, x_dim)
    }
    central_params = {
        "hidden_dim": central_dim, "latent_dim": args.latent_dim,
        "norm": True, "dropout": args.dropout, "beta": args.beta,
    }
    classif_params = {
        "n_class": num_classes, "lambda": args.lambda_classif,
        "hidden_layers": classifier_dim, "dropout": args.dropout,
    }
    surv_params = {
        "lambda": args.lambda_survival, "dims": survival_dim,
        "activation": "SELU", "l2_reg": 1e-2,
        "norm": True, "dropout": args.dropout,
    }
    train_params = {"switch": args.p2_switch, "lr": args.lr}

    metrics = []
    for split in range(1, 6):
        samples_train, samples_val, samples_test = get_splits(args.cohorts, split)
        omics_train = get_sub_omics_df(omics_df, samples_train)
        omics_val = get_sub_omics_df(omics_df, samples_val)
        omics_test = get_sub_omics_df(omics_df, samples_test)

        model = CustOMICS(
            source_params=source_params, central_params=central_params,
            classif_params=classif_params, surv_params=surv_params,
            train_params=train_params, device=device,
        )
        model.fit(
            omics_train=omics_train, clinical_df=clinical_df,
            label=label_col, event=event_col, surv_time=surv_time_col,
            omics_val=omics_val, batch_size=args.batch_size,
            n_epochs=args.epochs, verbose=True,
        )
        metric = model.evaluate(
            omics_test=omics_test, clinical_df=clinical_df,
            label=label_col, event=event_col, surv_time=surv_time_col,
            task=args.task, batch_size=1024,
        )
        metrics.append(metric)
        print(f"Split {split}: {metric}")

    print(f"\nMean metric: {np.mean(metrics):.4f} ± {np.std(metrics):.4f}")
    return metrics


if __name__ == "__main__":
    args = _parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device != "cpu"
        else torch.device("cpu")
    )
    sources = args.sources.split(",")
    hidden_dim = [int(x) for x in args.hidden_dim.split(",")]
    central_dim = [int(x) for x in args.central_dim.split(",")]
    classifier_dim = [int(x) for x in args.classifier_dim.split(",")]
    survival_dim = [int(x) for x in args.survival_dim.split(",")]

    train(args, device, sources, hidden_dim, central_dim, classifier_dim, survival_dim)
