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
import pandas as pd
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
from art.estimators.classification import KerasClassifier
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_dataset(name:str='unswnb15'): 
    """Wrapper for loading the training and testing datasets. This script will do all of 
    the preprocessing and only return Numpy arrays with the data/labels. 

    :param name: String with the name of the dataset [unswnb15]
    """

    if name == 'unswnb15': 
        X_tr, y_tr, X_te, y_te = load_unswnb()
    elif name == 'nslkdd': 
        X_tr, y_tr, X_te, y_te = load_nslkdd()
        
    return X_tr, y_tr, X_te, y_te


def load_nslkdd():
    """Load the NSL-KDD dataset from the data/ folder. Note you need to download the data 
    and add it to the folder. 

    :return Four Numpy arrays with X_tr, y_tr, X_te and y_te
    """

    df_tr = pd.read_csv('data/KDDTrain+.arff')
    df_te = pd.read_csv('data/KDDTest+.arff')
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})

    # change the name of the label column. this needs be be done if we are going to feed it into the 
    # data frame standardizer 
    df_tr['target'][df_tr['target']=='normal'] = 1
    df_tr['target'][df_tr['target']=='anomaly'] = -1

    df_te['target'][df_te['target']=='normal'] = 1
    df_te['target'][df_te['target']=='anomaly'] = -1

    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)

    X_tr, y_tr = df_tr.values[:,:-1], df_tr['target'].values
    X_te, y_te = df_te.values[:,:-1], df_te['target'].values

    return X_tr, y_tr, X_te, y_te


def load_unswnb(): 
    """Load the UNDWNB15 dataset from the data/ folder. Note you need to download the data 
    and add it to the folder. 

    :return Four Numpy arrays with X_tr, y_tr, X_te and y_te
    """
    drop_cols = ['id', 'proto', 'service', 'state', 'attack_cat', 'is_sm_ips_ports']

    df_tr = pd.read_csv('data/UNSW_NB15_training-set.csv')
    df_te = pd.read_csv('data/UNSW_NB15_testing-set.csv')
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"label": "target"}).drop(drop_cols, axis = 1)
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    X_tr, y_tr = df_tr.values[:,:-1], df_tr['target'].values
    X_te, y_te = df_te.values[:,:-1], df_te['target'].values

    # change the labes; 1=normal; -1=maliicious
    y_tr[y_tr==1] = -1
    y_tr[y_tr==0] = 1
    y_te[y_te==1] = -1
    y_te[y_te==0] = 1


    return X_tr, y_tr, X_te, y_te


def standardize_df_off_tr(df_tr:pd.DataFrame, df_te:pd.DataFrame): 
    """Standardize dataframes from a training and testing frame, where the means and 
    standard deviations that are calculated from the training dataset. 
    
    :param df_tr: Pandas dataframe with the training data 
    :param df_te: Pandas dataframe with the testing data 
    :return Two dataframes df_tr and df_te that have been standardized 
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
                              X:np.ndarray, 
                              ctype:str='svc', 
                              atype:str='fgsm'):
    """Generate adversarial data samples 

    :param X_tr: Dataset 
    :param y_tr: Labels vector 
    :param X: Dataset that is the seed for the adversarial samples
    :param ctype: String that is the classifier used to generate the attack ['svc', 'dt', 'mlp']
    :param atype: String that is the attack type ['fgsm', 'dt', 'deepfool', 'cw', 'pgd']
    :return Numpy array with the adversarial attack dataset
    """
    if ctype == 'svc': 
        clfr = SVC(C=1.0, kernel='rbf')
    elif ctype == 'gbc':
        # currently does not work.  
        clfr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    elif ctype == 'dt': 
        clfr = DecisionTreeClassifier(criterion='gini', 
                                      splitter='best', 
                                      max_depth=10, 
                                      min_samples_split=6, 
                                      min_samples_leaf=4) 
    elif ctype == 'mlp': 
        X_train, Y_train = X_tr, y_tr
        Y_train = tf.keras.utils.to_categorical(Y_train, 2)
        
        input_shape = (X_tr.shape[1],)
        num_classes = 2
        
        clfr = Sequential()
        clfr.add(Dense(128, input_shape=input_shape, activation='relu'))
        clfr.add(Dense(128, activation='relu'))
        clfr.add(Dropout(0.2, input_shape=(128,)))
        clfr.add(Dense(64, activation='relu'))
        clfr.add(Dense(num_classes, activation='softmax'))

        # Configure the model and start training
        clfr.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
        clfr.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.2)
        clfr = KerasClassifier(model=clfr, clip_values=(-5, 5), use_logits=False)
    else: 
        raise ValueError('Unknown classifier was set to generate the adversarial attacks.')
    
    if ctype == 'svc' or ctype == 'dt': 
        ytr_ohe = tf.keras.utils.to_categorical(y_tr, 2)
        clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
        clfr.fit(X_tr, ytr_ohe)


    if atype == 'fgsm': 
        attack = FastGradientMethod(estimator=clfr, eps=.2)
    elif atype == 'deepfool': 
        attack = DeepFool(clfr, verbose=False)
    elif atype == 'cw': 
        attack = CarliniL2Method(classifier=clfr, targeted=False, verbose=False)
    elif atype == 'pgd': 
        attack = ProjectedGradientDescent(clfr, eps=1.0, eps_step=0.1, verbose=False)
    elif atype == 'dt': 
        if ctype != 'dt': 
            raise ValueError('ctype and atype must both be decision trees for the attack and classifier when one is called.')
        attack = DecisionTreeAttack(clfr)
    Xadv = attack.generate(x=X)
    return Xadv 
