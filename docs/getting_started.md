## Installing CustOmics

CustOmics can be installed from `PyPI` on all OS, for any Python version `>=3.11`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager.

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name customics python=3.12
    conda activate customics
    ```

Choose one of the following, depending on your needs:

=== "From PyPI"

    ```sh
    pip install customics
    ```

=== "Editable mode"

    ``` bash
    git clone https://github.com/HakimBenkirane/CustOmics.git
    cd sopa

    pip install  -e .
    ```

=== "uv (dev mode)"

    ``` bash
    git clone https://github.com/HakimBenkirane/CustOmics.git
    cd customics

    uv sync --dev
    ```

## Usage
`CostOmics` provides a simple, scikit-learn–style `API` through the customics package. Below is a complete example showing how to train, evaluate, and interpret a multi-omics model:

#### 1. Prepare your data
`omics_train`: *dict mapping source name → pd.DataFrame (samples × features)*<br>
`clinical_df`: *pd.DataFrame with columns for labels, event indicator, and survival time.*


``` python
import torch
import pandas as pd
from customics import CustOMICS

omics_train = {
    "rna":   pd.read_csv("rna_train.csv",   index_col=0),
    "cnv":   pd.read_csv("cnv_train.csv",   index_col=0),
    "methyl": pd.read_csv("methyl_train.csv", index_col=0),
}
clinical_df = pd.read_csv("clinical.csv", index_col=0)
```

#### 2. Configure the model

``` python
source_params = {
    "rna":   {"input_dim": 5000, "hidden_dim": [1024, 512], "latent_dim": 128, "norm": True, "dropout": 0.2},
    "cnv":   {"input_dim": 2000, "hidden_dim": [512, 256],  "latent_dim": 128, "norm": True, "dropout": 0.2},
    "methyl":{"input_dim": 8000, "hidden_dim": [1024, 512], "latent_dim": 128, "norm": True, "dropout": 0.2},
}
central_params = {"hidden_dim": [512, 256], "latent_dim": 128, "norm": True, "dropout": 0.2, "beta": 1}
classif_params = {"n_class": 5, "lambda": 5.0, "hidden_layers": [128, 64], "dropout": 0.2}
surv_params    = {"lambda": 1.0, "dims": [64, 32], "activation": "SELU",
                  "l2_reg": 1e-2, "norm": True, "dropout": 0.2}
train_params   = {"switch": 10, "lr": 1e-3}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

#### 3. Train

``` python
model = CustOMICS(
    source_params=source_params,
    central_params=central_params,
    classif_params=classif_params,
    surv_params=surv_params,
    train_params=train_params,
    device=device,
)
model.fit(
    omics_train=omics_train,
    clinical_df=clinical_df,
    label="PAM50",       # classification target column
    event="OS",          # survival event column (0/1)
    surv_time="OS.time", # survival time column
    omics_val=omics_val, # optional validation set
    batch_size=32,
    n_epochs=30,
    verbose=True,
)
```

#### 4. Evaluate
Classification metrics (Accuracy, F1, AUC, …)

``` python
metrics = model.evaluate(
    omics_test, clinical_df,
    label="PAM50", event="OS", surv_time="OS.time",
    task="classification",
)
# Survival concordance index
ci = model.evaluate(
    omics_test, clinical_df,
    label="PAM50", event="OS", surv_time="OS.time",
    task="survival",
)
```

#### 4. Visualise & explain

``` python
model.plot_loss()
model.plot_representation(omics_train, clinical_df, label="PAM50",
                          filename="latent_space", title="t-SNE of latent space")
model.stratify(omics_train, clinical_df, event="OS", surv_time="OS.time")
model.explain(sample_ids, omics_train, clinical_df,
              source="rna", subtype="Her2", label="PAM50")
```

See the tutorial [here](../tutorials/usage) for more.
