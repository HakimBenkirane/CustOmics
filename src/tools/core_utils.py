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



def omics_train_test_split(omics_df, lt_samples_train, lt_samples_test):
    omics_train = {}
    omics_test = {}
    for name in omics_df.keys():
        omics_train[name] = omics_df[name].loc[lt_samples_train, :]
        omics_test[name] = omics_df[name].loc[lt_samples_test, :]
    return omics_train, omics_test

def train(task, cohort, sources, split, device, num_classes=4, batch_size=32, n_epochs=5, beta=1, lr=1e-3,
            hidden_dim=[512,256], central_dim=[512, 256], latent_dim=128, dropout=0.2,
            classifier_dim=[256,128], survival_dim=[64,32], lambda_classif=0, lambda_survival=0,
            explain=False, explained_source='RNAseq', explained_class='Her2'):


    omics_df, clinical_df, dataset_train, dataset_val, dataset_test, lt_samples, x_dim, ohe, le = prepare_dataset(cohort, sources, split)


    kwargs = {'num_workers': 2, 'pin_memory': True} if device.type == "cuda" else {}
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
    val_loader = DataLoader(dataset_val, batch_size=1024, shuffle=False, **kwargs)
    test_loader = DataLoader(dataset_test, batch_size=10024, shuffle=False, **kwargs)

    n_source = len(sources)

    if task == 'classification':
        classifier = MultiClassifier(n_class=num_classes, latent_dim=latent_dim, dropout=dropout,
            class_dim = classifier_dim).to(device)
    else:
        classifier = None
    if task == 'survival':
        surv_params = {'drop': 0.16, 'norm': True, 'dims': [latent_dim] + survival_dim + [1], 'activation': 'SELU', 'l2_reg':1e-2, 'device': device}
        survival_predictor = SurvivalNet(surv_params)
    else:
        survival_predictor = None

    lt_encoders = []
    lt_decoders = []
    for i in range(n_source):
        lt_encoders.append(Encoder(input_dim=x_dim[i], hidden_dim=hidden_dim, latent_dim=latent_dim, dropout=dropout))
        lt_decoders.append(Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim[i], dropout=dropout))
    central_encoder = ProbabilisticEncoder(input_dim = n_source * latent_dim, hidden_dim=central_dim, latent_dim=latent_dim, dropout=dropout)
    central_decoder = ProbabilisticDecoder(latent_dim=latent_dim, hidden_dim=central_dim, output_dim=n_source * latent_dim, dropout=dropout)

    samples_train, samples_test = dataset_train.return_samples(), dataset_test.return_samples()

    omics_train, omics_test = omics_train_test_split(omics_df, samples_train, samples_test)
    model = CustOMICS(n_source, lt_encoders, lt_decoders, central_encoder, central_decoder, device, 
                      beta = beta, lr=lr, num_classes=num_classes, classifier=classifier, survival_predictor=survival_predictor, switch=min(5, n_epochs - 1), 
                      lambda_classif=lambda_classif, lambda_survival=lambda_survival).to(device)

    model.fit(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, verbose=True)
    if task == 'classification':
        classification_metrics = model.evaluate(test_loader, ohe)
        print(classification_metrics)
        if explain:
            explain_vae(lt_samples, model, omics_df[explained_source], clinical_df, explained_source, explained_class, device, le)
        model.plot_representation(omics_train, clinical_df['PAM50'], samples_train, 'representation', title='Vizualization of the latent representation for the TCGA-BRCA cohort', le=le)
        return classification_metrics
    if task == 'survival':
        c_index = model.evaluate(test_loader, ohe)
        model.stratify(omics_test, samples_test, clinical_df, cohort)
        return c_index