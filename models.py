#!/usr/bin/env python

# Copyright 2021 Gregory Ditzler 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import copy 
import numpy as np 
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from utils import get_performance
from concurrent.futures import ProcessPoolExecutor

def evaluate_models_exploratory(X_normal:np.ndarray, 
                                X_te:np.ndarray, 
                                X_adv_deepfool:np.ndarray, 
                                X_adv_fgsm:np.ndarray, 
                                X_adv_pgd:np.ndarray, 
                                X_adv_dt:np.ndarray, 
                                Y:np.ndarray,
                                Y_aml:np.ndarray,  
                                perfs:dict, 
                                contamination:float=.05, 
                                degree:float=3., 
                                support_fraction:float=.5): 
    """
    """

    MODELS = [IsolationForest(contamination=contamination), 
              OneClassSVM(kernel='poly', degree=degree), 
              EllipticEnvelope(contamination=contamination, support_fraction=support_fraction),
              LocalOutlierFactor(contamination=contamination)]
    MODELS_NAMES = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['baseline', 'deepfool', 'fgsm', 'pgd', 'dt']

    for model, model_name in zip(MODELS, MODELS_NAMES): 
        # fit the model on the normal data 
        model.fit(X_normal)
        
        # if we are running the local outlier factor then we need to set the novelty bit 
        # in the class 
        if hasattr(model, 'novelty'): 
            model.novelty = True

        #Y_hat, Y_deepfool, Y_fgsm, Y_pgd, Y_dt 
        outputs = model.predict(X_te), model.predict(X_adv_deepfool), \
            model.predict(X_adv_fgsm), model.predict(X_adv_pgd), model.predict(X_adv_dt)

        for y_hat, attack_type in zip(outputs, ATTACKS): 
            if attack_type == 'baseline': 
                labels = Y
            else: 
                labels = Y_aml

            acc, fs, tpr, tnr, mcc = get_performance(y_true=labels, y_hat=y_hat)
            perfs[''.join(['accs_', model_name, '_', attack_type])] += acc
            perfs[''.join(['fss_', model_name, '_', attack_type])] += fs
            perfs[''.join(['tprs_', model_name, '_', attack_type])] += tpr
            perfs[''.join(['tnrs_', model_name, '_', attack_type])] += tnr
            perfs[''.join(['mccs_', model_name, '_', attack_type])] += mcc
    
    return perfs


def evaluate_models_causative(X:np.ndarray, 
                              X_pattern:np.ndarray, 
                              X_single:np.ndarray, 
                              X_svc:np.ndarray,
                              X_te:np.ndarray, 
                              Y_te:np.ndarray,  
                              perfs:dict, 
                              contamination:float=.05, 
                              degree:float=3., 
                              support_fraction:float=.5):
    """
    """
    MODELS = [IsolationForest(contamination=contamination), 
              OneClassSVM(kernel='poly', degree=degree), 
              EllipticEnvelope(contamination=contamination, support_fraction=support_fraction),
              LocalOutlierFactor(contamination=contamination)]
    MODELS_NAMES = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['baseline', 'pattern', 'single', 'svc']

    for model, model_name in zip(MODELS, MODELS_NAMES): 
        model_n = copy.copy(model).fit(X)
        model_pattern = copy.copy(model).fit(X_pattern)
        model_single = copy.copy(model).fit(X_single)
        model_svc = copy.copy(model).fit(X_svc)

        if hasattr(model_n, 'novelty'): 
            model_n.novelty = True
            model_pattern.novelty = True 
            model_single.novelty = True
            model_svc.novelty = True 
        
        # y_lo, y_lo_pattern, y_lo_single, y_lo_svc = 
        outputs = model_n.predict(X_te), model_pattern.predict(X_te), model_single.predict(X_te), model_svc.predict(X_te)
        
        for y_hat, attack_type in zip(outputs, ATTACKS): 

            acc, fs, tpr, tnr, mcc = get_performance(y_true=Y_te, y_hat=y_hat)
            perfs[''.join(['accs_', model_name, '_', attack_type])] += acc
            perfs[''.join(['fss_', model_name, '_', attack_type])] += fs
            perfs[''.join(['tprs_', model_name, '_', attack_type])] += tpr
            perfs[''.join(['tnrs_', model_name, '_', attack_type])] += tnr
            perfs[''.join(['mccs_', model_name, '_', attack_type])] += mcc

    return perfs 