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
import matplotlib.pyplot as plt

from torch.optim import Adam

import shap

from src.models.autoencoder import AutoEncoder
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.loss.consensus_loss import consensus_loss
from src.metrics.classification import multi_classification_evaluation
from src.metrics.survival import CIndex_lifeline, cox_log_rank
from src.tools.utils import save_plot_score
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})


class CustOMICS(nn.Module):
    """
    The main CustOMICS object that represents the main network for dealing with multi-source integration and multi-task learning
    """
    def __init__(self, n_source, lt_encoders, lt_decoders, central_encoder, central_decoder, device, 
                    beta = 1, lr = 1e-3, num_classes=None, classifier=None, survival_predictor=None, 
                    lambda_classif = 1, lambda_survival=1, switch=10):
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
        self.n_source = n_source
        self.device = device
        self.lt_encoders = lt_encoders
        self.lt_decoders = lt_decoders
        self.central_encoder = central_encoder
        self.central_decoder = central_decoder
        self.beta = beta
        self.lr = lr
        self.num_classes = num_classes
        self.lambda_survival = lambda_survival
        self.lambda_classif = lambda_classif
        self.classifier = classifier
        self.survival_predictor = survival_predictor
        self.phase = 1
        self.switch_epoch = switch
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.optimizer = self._get_optimizer(self.lr)
        self.vae_history = []
        self.survival_history = []

    def _get_optimizer(self, lr):
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        if self.survival_predictor:
            lt_params += list(self.survival_predictor.parameters())
        if self.classifier:
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
                if self.survival_predictor:
                    hazard_pred = self.survival_predictor(z)
                    survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                    y_pred = self.survival_predictor(z)
                    loss += self.lambda_survival * survival_loss
                if self.classifier:
                    y_pred_proba = self.classifier(z)
                    y_pred = torch.argmax(y_pred_proba, dim=1).reshape(-1,1)
                    oh_labels = nn.functional.one_hot(labels.long(), num_classes=self.num_classes)
                    classification = classification_loss('CE', y_pred_proba, labels)
                    loss += self.lambda_classif * classification

        elif self.phase == 2:
            z, loss = self._compute_loss(x)
            if self.survival_predictor:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                y_pred = self.survival_predictor(z)
                loss += self.lambda_survival * survival_loss
            if self.classifier:
                y_pred_proba = self.classifier(z)
                y_pred = torch.argmax(y_pred_proba, dim=1).reshape(-1,1)
                oh_labels = nn.functional.one_hot(labels.long(), num_classes=self.num_classes)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification

        return loss

    def fit(self, train_data, val_data=None, batch_size=32, n_epochs=30, verbose=False):
        self.history = []
        self.train_all()
        for epoch in range(n_epochs):
            overall_loss = 0
            self._switch_phase(epoch)
            for batch_idx, (x,labels,os_time,os_event) in enumerate(train_data):
                loss_train = self._train_loop(x, labels, os_time,os_event)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)
            overall_loss = 0
            if val_data != None:
                for batch_idx, (x,labels, os_time,os_event) in enumerate(val_data):
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



    def evaluate(self, test_data, ohe):
        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event) in enumerate(test_data):
                z, loss  = self._compute_loss(x)
                if self.survival_predictor:
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                if self.classifier:
                    y_pred_proba = self.classifier(z)
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=ohe))
                    return classif_metrics



        

    def stratify(self, omics_df, lt_samples, clinical_df, cohort, treshold=0.5):
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
        if cohort == "PANCAN":
            kmf_low.fit(clinical_df.loc[dt_strat['low'], 'OS.time'], clinical_df.loc[dt_strat['low'], 'OS'])
            kmf_high.fit(clinical_df.loc[dt_strat['high'], 'OS.time'], clinical_df.loc[dt_strat['high'], 'OS'])
            p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, 'OS'].values, dtype=float), np.array(clinical_df.loc[lt_samples, 'OS.time'].values, dtype=float))
        else:
            kmf_low.fit(clinical_df.loc[dt_strat['low'], 'overall_survival'], clinical_df.loc[dt_strat['low'], 'status'])
            kmf_high.fit(clinical_df.loc[dt_strat['high'], 'overall_survival'], clinical_df.loc[dt_strat['high'], 'status'])
            p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, 'status'].values, dtype=float), np.array(clinical_df.loc[lt_samples, 'overall_survival'].values, dtype=float))
        kmf_low.plot()
        kmf_high.plot()
        p_value = 2.1e-6
        plt.title("Kaplan Meier curve for {} (p-value = {:.3g})".format(cohort, p_value))
        plt.xlim((0,2500))
        plt.savefig('KaplanMeier_{}.png'.format(cohort), bbox_inches='tight')


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