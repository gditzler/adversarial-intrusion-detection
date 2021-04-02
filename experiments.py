#!/usr/bin/env python



import numpy as np 

from data import load_dataset, generate_adversarial_data
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest


def run_experiment(dataset:str='unswnb15', 
                   ids:str='isofor', 
                   trials:int=10, 
                   verbose:bool=False): 
    """
    """

    if verbose: 
        print(''.join(['Dataset: ', dataset]))

    # load the dataset and split by training/testing data 
    X_tr, y_tr, X_te, y_te = load_dataset(dataset)
    # get the size of the tr/te data 
    n_tr, n_te = len(X_tr), len(X_te)

    # set up a cv partition over the number fo cv trials. 
    kf = KFold(n_splits=trials)
    kf.get_n_splits(X_tr)

    t = 0
    for train_index, test_index in kf.split(X_tr):
        if verbose:
            print(''.join([' -> Running round ', str(t+1), ' of ', str(trials)]))

        # split the data out by the training and validation indices 
        X_tr_t, y_tr_t, X_a_t, y_a_t = X_tr[train_index,:], y_tr[train_index], \
            X_tr[test_index,:], y_tr[test_index]

        X_adv_t = generate_adversarial_data(X_tr=X_tr_t, y_tr=y_tr_t, X=X_a_t)



        t += 1 


    return None 