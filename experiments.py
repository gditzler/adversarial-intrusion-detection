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

import numpy as np 

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import KFold

from utils import get_performance


def run_experiment_exploratory(dataset:str='unswnb15', 
                               trials:int=10, 
                               verbose:bool=False): 
    """Run the experiment for exploratory attacks against intrusion detection. 
    """

    if verbose: 
        print(''.join(['Dataset: ', dataset]))
    
    n_attacks = 4
    n_merits = 4
    support_fraction = .5
    contamination = .05
    degree = 3


    
    # load the data from the npz files. note that all of the X_tr, X_te, y_tr and y_te are the same 
    # regarless of the file. the difference is in how the Xaml data are generated from a MLPNN. the 
    # labels of y_te are the initial labels of the adversarial data. 
    data = np.load(''.join(['data/full_data_', dataset, '_dt_dt.npz']))
    X_tr, y_tr, X_te, y_te, X_adv_dt = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 
    
    # lad the deepfool data 
    data = np.load(''.join(['data/full_data_', dataset, '_mlp_deepfool.npz']))
    _, _, _, _, X_adv_deepfool = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml']

    # load the fgsm data 
    data = np.load(''.join(['data/full_data_', dataset, '_mlp_fgsm.npz']))
    _, _, _, _, X_adv_fgsm = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 

    # load the pgd data 
    data = np.load(''.join(['data/full_data_', dataset, '_mlp_pgd.npz']))
    _, _, _, _, X_adv_pgd = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml']  

    # we need to set up the k-fold evaluator 
    kf = KFold(n_splits=trials)

    for train_index, _ in kf.split(X_tr):
        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        X_tr_n, y_tr_n = X_tr[train_index,:], y_tr[train_index]

        # set the normal data 
        X_tr_n_normal = X_tr_n[y_tr_n == 0]

        # isolation forest 
        model = IsolationForest(contamination=contamination).fit(X_tr_n_normal)
        y_if = model.predict(X_te)
        y_if_deepfool = model.predict(X_adv_deepfool)
        y_if_fgsm = model.predict(X_adv_fgsm)
        y_if_pgd = model.predict(X_adv_pgd)
        y_if_dt = model.predict(X_adv_dt)

        # osvm 
        model = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_normal)
        y_svm = model.predict(X_te)
        y_svm_deepfool = model.predict(X_adv_deepfool)
        y_svm_fgsm = model.predict(X_adv_fgsm)
        y_svm_pgd = model.predict(X_adv_pgd)
        y_svm_dt = model.predict(X_adv_dt)

        # ellicptic 
        model = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_normal)
        y_ee = model.predict(X_te)
        y_ee_deepfool = model.predict(X_adv_deepfool)
        y_ee_fgsm = model.predict(X_adv_fgsm)
        y_ee_pgd = model.predict(X_adv_pgd)
        y_ee_dt = model.predict(X_adv_dt)

        # local outliers
        model = LocalOutlierFactor(contamination=0.05).fit(X_tr_n_normal)
        model.novelty = True

        y_lo = model.predict(X_te)
        y_lo_deepfool = model.predict(X_adv_deepfool)
        y_lo_fgsm = model.predict(X_adv_fgsm)
        y_lo_pgd = model.predict(X_adv_pgd)
        y_lo_dt = model.predict(X_adv_dt)


        acc_if_deepfool, fs_if_deepfool, tpr_if_deepfool, tnr_if_deepfool, mcc_if_deepfool = get_performance(y_true=y_te, y_hat=y_if_deepfool)
        acc_svm_deepfool, fs_svm_deepfool, tpr_svm_deepfool, tnr_svm_deepfool, mcc_svm_deepfool = get_performance(y_true=y_te, y_hat=y_svm_deepfool)
        acc_ee_deepfool, fs_ee_deepfool, tpr_ee_deepfool, tnr_ee_deepfool, mcc_ee_deepfool = get_performance(y_true=y_te, y_hat=y_ee_deepfool)
        acc_lo_deepfool, fs_lo_deepfool, tpr_lo_deepfool, tnr_lo_deepfool, mcc_lo_deepfool = get_performance(y_true=y_te, y_hat=y_lo_deepfool)

        acc_if_fgsm, fs_if_fgsm, tpr_if_fgsm, tnr_if_fgsm, mcc_if_fgsm = get_performance(y_true=y_te, y_hat=y_if_fgsm)
        acc_svm_fgsm, fs_svm_fgsm, tpr_svm_fgsm, tnr_svm_fgsm, mcc_svm_fgsm = get_performance(y_true=y_te, y_hat=y_svm_fgsm)
        acc_ee_fgsm, fs_ee_fgsm, tpr_ee_fgsm, tnr_ee_fgsm, mcc_ee_fgsm = get_performance(y_true=y_te, y_hat=y_ee_fgsm)
        acc_lo_fgsm, fs_lo_fgsm, tpr_lo_fgsm, tnr_lo_fgsm, mcc_lo_fgsm = get_performance(y_true=y_te, y_hat=y_lo_fgsm)

        acc_if_pgd, fs_if_pgd, tpr_if_pgd, tnr_if_pgd, mcc_if_pgd = get_performance(y_true=y_te, y_hat=y_if_pgd)
        acc_svm_pgd, fs_svm_pgd, tpr_svm_pgd, tnr_svm_pgd, mcc_svm_pgd = get_performance(y_true=y_te, y_hat=y_svm_pgd)
        acc_ee_pgd, fs_ee_pgd, tpr_ee_pgd, tnr_ee_pgd, mcc_ee_pgd = get_performance(y_true=y_te, y_hat=y_ee_pgd)
        acc_lo_pgd, fs_lo_pgd, tpr_lo_pgd, tnr_lo_pgd, mcc_lo_pgd = get_performance(y_true=y_te, y_hat=y_lo_pgd)

        acc_if_dt, fs_if_dt, tpr_if_dt, tnr_if_dt, mcc_if_dt = get_performance(y_true=y_te, y_hat=y_if_dt)
        acc_svm_dt, fs_svm_dt, tpr_svm_dt, tnr_svm_dt, mcc_svm_dt = get_performance(y_true=y_te, y_hat=y_svm_dt)
        acc_ee_dt, fs_ee_dt, tpr_ee_dt, tnr_ee_dt, mcc_ee_dt = get_performance(y_true=y_te, y_hat=y_ee_dt)
        acc_lo_dt, fs_lo_dt, tpr_lo_dt, tnr_lo_dt, mcc_lo_dt = get_performance(y_true=y_te, y_hat=y_lo_dt)





    

    return None 