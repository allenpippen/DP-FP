#!/usr/bin/python
# -*- coding:utf8 -*-
from scipy.io import loadmat,savemat
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score


def load_data_kfold(k,datatype,path,random_state=111):
    data = loadmat(path)
    if datatype == 'sfc':
        X = data.get('fnc')
    elif datatype == 'tc':
        X = data.get('tc')
    y = data.get('label').T
    # print(X)
    # print(y)
    folds = list(StratifiedShuffleSplit(n_splits=k,random_state=random_state).split(X, y))
    # print(folds)
    return folds, X, y

from sklearn.metrics import auc
from sklearn import metrics
def acc_pre_recall_f(y_true,y_pred,y_score):
    #print(y_true)
    #print(y_true.shape)
    #print(y_pred.reshape(1, 10))
    #print(y_pred.shape)
    # print(y_true, y_pred, y_score)
    tp = float(sum(y_true == y_pred))
    fp = float(sum((y_true == 0) & (y_pred == 1)))
    tn = float(sum((y_true == 0) & (y_pred == 0)))
    fn = float(sum((y_true == 1) & (y_pred == 0)))
    acc = accuracy_score(y_true,y_pred)
    sensitivity = tp/(tp + fn)
    f1 = f1_score(y_true,y_pred)
    specificity = tn / (fp + tn)
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_score)
    roc_auc = auc(fpr, tpr)
    return acc,specificity,sensitivity,f1,roc_auc

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Val loss score has improved from {} to {}!".format(self.best_score, score))
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

