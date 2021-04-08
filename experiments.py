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
    """
    """

    if verbose: 
        print(''.join(['Dataset: ', dataset]))
    
    n_attacks = 4
    n_merits = 4


    
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

    

    return None 