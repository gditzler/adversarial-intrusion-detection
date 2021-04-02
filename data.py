#!/usr/bin/env python 
import numpy as np 
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVC

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from art.estimators.classification.scikitlearn import ScikitlearnSVC


def load_dataset(name:str='unswnb15'): 
    """
    """

    if name == 'unswnb15': 
        X_tr, y_tr, X_te, y_te = load_unswnb()
        
    return X_tr, y_tr, X_te, y_te



def load_unswnb(): 
    # we need to drop these columns from the data
    drop_cols = ['id', 'proto', 'service', 'state', 'attack_cat', 'is_sm_ips_ports']

    df_tr = pd.read_csv('data/UNSW_NB15_training-set.csv')
    df_te = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    X_tr, y_tr = df_tr.values[:,:-1], df_tr['target'].values
    X_te, y_te = df_te.values[:,:-1], df_te['target'].values
    return X_tr, y_tr, X_te, y_te


def standardize_df_off_tr(df_tr:pd.DataFrame, df_te:pd.DataFrame): 
    """
    Standardize dataframes from a training and testing frame, where the means
    and standard deviations that are calculated from the training dataset. 
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    """
    for key in df_tr.keys(): 
        if key != 'target': 
            # scale the testing data w/ the training means/stds
            df_te[key] = (df_te[key].values - df_tr[key].values.mean())/df_tr[key].values.std()
            # scale the training data 
            df_tr[key] = (df_tr[key].values - df_tr[key].values.mean())/df_tr[key].values.std()
    return df_tr, df_te


def generate_adversarial_data(X_tr:np.ndarray, 
                              y_tr:np.ndarray, 
                              X:np.ndarray):
    clfr = SVC(C=1.0, kernel='rbf')
    ytr_ohe = tf.keras.utils.to_categorical(y_tr, 2)
    clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
    clfr.fit(X_tr, ytr_ohe)
    attack = FastGradientMethod(estimator=clfr, eps=.2)
    Xadv = attack.generate(x=X)
    return Xadv 
