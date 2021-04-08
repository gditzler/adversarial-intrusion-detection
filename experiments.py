#!/usr/bin/env python



import numpy as np 

from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor


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
        y_lo = model.predict(X_te)
        y_lo_deepfool = model.predict(X_adv_deepfool)
        y_lo_fgsm = model.predict(X_adv_fgsm)
        y_lo_pgd = model.predict(X_adv_pgd)
        y_lo_dt = model.predict(X_adv_dt)


    

    return None 