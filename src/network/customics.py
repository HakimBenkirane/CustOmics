# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the CustOMICS module.
"""
import numpy as np

from src.loss.survival_loss import CoxLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from torch.optim import Adam

import shap

from src.datasets.multi_omics_dataset import MultiOmicsDataset
from src.models.autoencoder import AutoEncoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.tasks.classification import MultiClassifier
from src.tasks.survival import SurvivalNet
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.loss.consensus_loss import consensus_loss
from src.metrics.classification import multi_classification_evaluation
from src.metrics.survival import CIndex_lifeline, cox_log_rank
from src.tools.utils import save_plot_score
from src.tools.utils import get_common_samples
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})


class CustOMICS(nn.Module):
    """
    The main CustOMICS object that represents the main network for dealing with multi-source integration and multi-task learning
    """
    def __init__(self, source_params, central_params, classif_params, surv_params, train_params, device):
        """
        Construct the whole architecture with intermediate autoencoders, central layer and eventually downstream predictors
        Parameters:
            n_source (int)            -- the number of sources to consider in input
            lt_encoders (list)        -- list of encoders leading from the input to the central layer's input
            lt_decoders (list)        -- list of decoders leading from the central layer's output to the model's output
            central_encoder (encoder) -- the encoder of the central layer
            central_decoder (encoder) -- the decoder of the central layer
            device (pytorch)          -- the device in which the computation is done
            beta (float)              -- the beta parameter of the VAE's regularization loss
            lr (float)                -- the learning rate for the optimizer
            predictor (Module)        -- the predictor for the eventual downstream task (if None, unsupervised setting)
            task (string)             -- downstream task to perform (if None, unsupervised setting)
            consensus (boolean)       -- parameter for the consensus loss
            variational (boolean)     -- whether or not to consider variational inference in intermediate autoencoders
        """
        super(CustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.device = device
        self.lt_encoders = [Encoder(input_dim=source_params[source]['input_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             latent_dim=source_params[source]['latent_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.lt_decoders = [Decoder(latent_dim=source_params[source]['latent_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             output_dim=source_params[source]['input_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.rep_dim = sum([source_params[source]['latent_dim'] for source in source_params])
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.beta = central_params['beta']
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier =  MultiClassifier(n_class=self.num_classes, latent_dim=central_params['latent_dim'], dropout=classif_params['dropout'],
            class_dim = classif_params['hidden_layers']).to(self.device)
        self.lambda_survival = surv_params['lambda']
        surv_param = {'drop': surv_params['dropout'], 'norm': surv_params['norm'], 'dims': [central_params['latent_dim']] + surv_params['dims'] + [1], 
                    'activation': surv_params['activation'], 'l2_reg': surv_params['l2_reg'], 'device': self.device}
        self.survival_predictor = SurvivalNet(surv_param)
        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.survival_history = []
        self.label_encoder = None
        self.one_hot_encoder = None

    def _get_optimizer(self, lr):
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        lt_params += list(self.survival_predictor.parameters())
        lt_params += list(self.classifier.parameters())        
        optimizer = Adam(lt_params, lr=lr)
        return optimizer

    def _set_autoencoders(self):
        for i in range(self.n_source):
            self.autoencoders.append(AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], self.device))

    def _set_central_layer(self):
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        for i in range(self.n_source):
            self.autoencoders[i].to(self.device)
        self.central_layer.to(self.device)

    def _switch_phase(self, epoch):
        if epoch < self.switch_epoch:
            self.phase = 1
        else:
            self.phase = 2


    def per_source_forward(self, x):
        lt_forward = []
        for i in range(self.n_source):
            lt_forward.append(self.autoencoders[i](x[i]))
        return lt_forward

    def get_per_source_representation(self, x):
        lt_rep = []
        for i in range(self.n_source):
            lt_rep.append(self.autoencoders[i](x[i])[1])
        return lt_rep

    def decode_per_source_representation(self, lt_rep):
        lt_hat = []
        for i in range(self.n_source):
            lt_hat.append(self.autoencoders[i].decode(lt_rep[i]))
        return lt_hat



    def forward(self, x):
        lt_forward = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_forward]
        lt_rep = [element[1] for element in lt_forward]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep ,mean

    def _compute_loss(self, x):
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            return lt_rep, loss
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for source, autoencoder in zip(x, self.autoencoders):
                loss += autoencoder.loss(source, self.beta)
            central_concat = torch.cat(lt_rep, dim=1)
            loss += self.central_layer.loss(central_concat, self.beta)
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss

    def _train_loop(self, x, labels, os_time, os_event):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        # If we don't disjoint the autoencoder's architecture 
        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x)
            for z in lt_rep:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                loss += self.lambda_survival * survival_loss
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification

        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            hazard_pred = self.survival_predictor(z)
            survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
            loss += self.lambda_survival * survival_loss
            y_pred_proba = self.classifier(z)
            classification = classification_loss('CE', y_pred_proba, labels)
            loss += self.lambda_classif * classification

        return loss

    def fit(self, omics_train, clinical_df, label, event, surv_time, omics_val=None, batch_size=32, n_epochs=30, verbose=False):
        
        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical_df.loc[:, label].values)
        encoded_clinical_df.loc[:, 'PAM50'] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)
        self.one_hot_encoder = OneHotEncoder(sparse=False).fit(encoded_clinical_df.loc[:, label].values.reshape(-1,1))

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])
        dataset_train = MultiOmicsDataset(omics_df=omics_train, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(omics_df=omics_val, clinical_df=encoded_clinical_df, lt_samples=lt_samples_val,
                                            label=label, event=event, surv_time=surv_time)
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)

        self.history = []
        self.train_all()
        for epoch in range(n_epochs):
            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x,labels,os_time,os_event) in enumerate(train_loader):
                loss_train = self._train_loop(x, labels, os_time,os_event)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)
            overall_loss = 0
            if val_loader != None:
                for batch_idx, (x,labels, os_time,os_event) in enumerate(val_loader):
                    loss_val = self._train_loop(x, labels, os_time,os_event)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx+1)*batch_size)

                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train, "\tAverage Loss Val : ", average_loss_val)
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train)


    def get_latent_representation(self, omics_df):
        self.eval_all()
        x = [torch.Tensor(omics_df[source].values) for source in omics_df.keys()]
        with torch.no_grad():
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
            z, loss = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def reconstruct(self, x):
        x = torch.Tensor(x)
        z = self.autoencoders[0](x)[1]
        return self.autoencoders[0].decode(z).cpu().detach().numpy()


    def plot_representation(self, omics_df, labels_df, lt_samples, source, title, le=None):
        if source == 'representation':
            z = self.get_latent_representation(omics_df=omics_df)
            save_plot_score('representation_plot', z, labels_df[lt_samples].values, title, le=le)
        else:
            print(labels_df)
            save_plot_score('{}_plot'.format(source), omics_df[source].values, labels_df[lt_samples].values, title, le=le)


    def source_predict(self, expr_df, source):
        #tensor_expr = torch.Tensor(expr_df.values)
        tensor_expr = expr_df
        if source == 'CNV':
            z = self.lt_encoders[0](tensor_expr)
        elif source == 'RNAseq':
            z = self.lt_encoders[1](tensor_expr)
        elif source == 'methyl':
            z = self.lt_encoders[2](tensor_expr)
        y_pred_proba = self.classifier(z)
        return y_pred_proba



    def evaluate(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32):

        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, 'PAM50'] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
                                            label=label, event=event, surv_time=surv_time)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_loader):
                z, loss  = self._compute_loss(x)
                print(z)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                elif task == 'classification':
                    y_pred_proba = self.classifier(z)
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                
                    return classif_metrics



    def stratify(self, omics_df, clinical_df, event, surv_time, treshold=0.5, 
                    save_plot=False, plot_title="", filename=''):
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df)
        hazard_pred = self.survival_predictor(torch.Tensor(z)).cpu().detach().numpy()
        dt_strat = {'high': [], 'low': []}
        for i in range(len(lt_samples)):
            if hazard_pred[i] <= np.mean(hazard_pred):
                dt_strat['low'].append(lt_samples[i])
            else:
                dt_strat['high'].append(lt_samples[i])
        kmf_low = KaplanMeierFitter(label='low risk')
        kmf_high = KaplanMeierFitter(label='high risk')
        kmf_low.fit(clinical_df.loc[dt_strat['low'], surv_time], clinical_df.loc[dt_strat['low'], event])
        kmf_high.fit(clinical_df.loc[dt_strat['high'], surv_time], clinical_df.loc[dt_strat['high'], event])
        p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, event].values, dtype=float), np.array(clinical_df.loc[lt_samples, surv_time].values, dtype=float))

        kmf_low.plot()
        kmf_high.plot()
        plt.title(plot_title + " (p-value = {:.3g})".format(p_value))
        plt.xlim((0,2500))
        if save_plot:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


    def explain(self, omics_df, source, patient):
        data_summary = shap.kmeans(omics_df[source], 100)
        explainer_autoencoder = shap.KernelExplainer(self.reconstruct, omics_df[source].values)
        shap_values = explainer_autoencoder.shap_values(omics_df[source].loc[patient,:].values)
        print(shap_values)

    def print_parameters(self):
        lt_params = []
        lt_names = []
        for autoencoder in self.autoencoders:
            for name, param in autoencoder.named_parameters():
                lt_params.appand(param.data)
                lt_names.append(name)
        for name, param in self.central_layer.named_parameters():
            lt_params.append(param.data)
            lt_names.append(name)
        print(len(lt_params))
        print(lt_names)

    def train_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.train()
            decoder.train()
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.central_layer.train()
        if self.survival_predictor:
            self.survival_predictor.train()
        if self.classifier:
            self.classifier.train()
    def eval_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.eval()
            decoder.eval()
        for autoencoder in self.autoencoders:
            autoencoder.eval()
        self.central_layer.eval()
        if self.survival_predictor:
            self.survival_predictor.eval()
        if self.classifier:
            self.classifier.eval()