from src.network.customics import CustOMICS
from src.tools.prepare_dataset import prepare_dataset
from src.tasks.classification import MultiClassifier
from src.tasks.survival import SurvivalNet
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.network.customics import CustOMICS
from src.ex_vae.shap_vae import explain_vae
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader


def train(task, cohorts, sources, split, device, num_classes=4,
             batch_size=32, n_epochs=10, beta=1, lr=1e-3,
            hidden_dim=[512,256], central_dim=[512,256], latent_dim=128, dropout=0.2,
            classifier_dim=[128, 64], survival_dim=[64,32], lambda_classif=5, 
            lambda_survival=5, explain=False, explained_source='RNAseq', explained_class='Her2'):


    omics_df, clinical_df, omics_train, omics_val, omics_test, lt_samples, x_dim = prepare_dataset(cohorts, sources, split)

    hidden_dim = [512, 256]
    central_dim = [512, 256]
    rep_dim = 128
    latent_dim=128

    source_params = {}
    central_params = {'hidden_dim': [128, 64], 'latent_dim': latent_dim, 'norm': True, 'dropout': 0.2, 'beta': 1}
    classif_params = {'n_class': 4, 'lambda': 5, 'hidden_layers': [128, 64], 'dropout': 0.2}
    surv_params = {'lambda': 20, 'dims': [64, 32], 'activation': 'SELU', 'l2_reg': 1e-2, 'norm': True, 'dropout': 0.2}
    for i, source in enumerate(sources):
        source_params[source] = {'input_dim': x_dim[i], 'hidden_dim': hidden_dim, 'latent_dim': rep_dim, 'norm': True, 'dropout': 0.2}
    train_params = {'switch': 5, 'lr': 1e-3}



    model = CustOMICS(source_params=source_params, central_params=central_params, classif_params=classif_params,
                            surv_params=surv_params, train_params=train_params, device=device).to(device)

    model.fit(omics_train=omics_train, clinical_df=clinical_df, label='PAM50', event='status', surv_time='overall_survival',
                omics_val=omics_val, batch_size=batch_size, n_epochs=n_epochs, verbose=True)
    metric = model.evaluate(omics_test=omics_test, clinical_df=clinical_df, label='PAM50', event='status', surv_time='overall_survival',
                    task='survival', batch_size=1024)

        #explain_vae(lt_samples, model, omics_df['RNAseq'], clinical_df, 'RNAseq', 'Her2', device, le)
    model.stratify(omics_df=omics_train, clinical_df=clinical_df, event='status', surv_time='overall_survival')
    return metric