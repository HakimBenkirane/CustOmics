# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Creates the different metrics to evaluate a classification task.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def roc_auc_score_multiclass(y_true, y_pred, ohe, average = "macro"):
    y_true = ohe.transform(np.array(y_true).reshape(-1,1))

    roc_auc = roc_auc_score(y_true, y_pred, average = average, multi_class='ovo')

    return roc_auc

def multi_classification_evaluation(y_true, y_pred, y_pred_proba, average='weighted', save_confusion=False, filename=None, plot_roc=False, ohe=None):
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average = average)
    f1_score = metrics.f1_score(y_true, y_pred, average = average)
    auc = roc_auc_score_multiclass(y_true, y_pred_proba, ohe, average = average)
    dt_scores = {'Accuracy': accuracy,
                'F1-score' : f1_score,
                'Precision' : precision,
                'Recall' : recall,
                'AUC' : auc} 

    if save_confusion:
        plt.figure(figsize = (18,8))
        sns.heatmap(metrics.confusion_matrix(y_true, y_pred), annot = True, xticklabels = np.unique(y_true), yticklabels = np.unique(y_true), cmap = 'summer')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig(filename + '.png')
        plt.clf()

    return dt_scores



    