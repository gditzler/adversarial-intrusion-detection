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
from sklearn.tree import DecisionTreeClassifier

from art.attacks.evasion import FastGradientMethod, DeepFool
from art.attacks.evasion.carlini import CarliniL2Method
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier
from art.estimators.classification import KerasClassifier
from art.attacks.evasion.decision_tree_attack import DecisionTreeAttack
from art.attacks.poisoning import PoisoningAttackSVM, PoisoningAttackCleanLabelBackdoor, PoisoningAttackBackdoor
from art.attacks.poisoning.perturbations import add_pattern_bd, add_single_bd


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
    elif name == 'awid': 
        X_tr, y_tr, X_te, y_te = load_awid()
        
    return X_tr, y_tr, X_te, y_te


def nslkddProtocolType(df_set:pd.DataFrame):
    df_set['protocol_type'][df_set['protocol_type'] == 'tcp'] = 0
    df_set['protocol_type'][df_set['protocol_type'] == 'udp'] = 1
    df_set['protocol_type'][df_set['protocol_type'] == 'icmp'] = 2
    return df_set


def nslkddService(df_set:pd.DataFrame):
    servicetypes = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf',
                    'daytime', 'discard', 'domain', 'domain_u', 'echo', 'eco_i',
                    'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data',
                    'gopher', 'harvest', 'hostnames', 'http', 'http_2784',
                    'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap',
                    'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp', 'name',
                    'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat',
                    'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2',
                    'pop_3', 'printer', 'private', 'red_i', 'remote_job',
                    'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc',
                    'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois',
                    'X11', 'Z39_50'
                    ]
    for i, servicename in enumerate(servicetypes):
        df_set['service'][df_set['service'] == servicename] = i
    return df_set


def nslkddFlag(df_set:pd.DataFrame):
    flagtypes = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2',
                 'S3', 'SF', 'SH'
                 ]
    for i, flagname in enumerate(flagtypes):
        df_set['flag'][df_set['flag'] == flagname] = i
    return df_set



def load_nslkdd():
    """Load the NSL-KDD dataset from the data/ folder. Note you need to download the data 
    and add it to the folder. 

    :return Four Numpy arrays with X_tr, y_tr, X_te and y_te
    """

    df_tr = pd.read_csv('data/NSLKDD/train.csv')
    df_te = pd.read_csv('data/NSLKDD/test.csv')
    df_tr = df_tr.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})
    df_te = df_te.sample(frac=1).reset_index(drop=True).rename(columns={"class": "target"})

    # change the name of the label column. this needs be be done if we are going to feed it into the 
    # data frame standardizer 
    df_tr['target'][df_tr['target']=='normal'] = 0
    df_tr['target'][df_tr['target']=='anomaly'] = 1

    df_te['target'][df_te['target']=='normal'] = 0
    df_te['target'][df_te['target']=='anomaly'] = 1

    df_tr, df_te = nslkddProtocolType(df_tr), nslkddProtocolType(df_te)
    df_tr, df_te = nslkddService(df_tr), nslkddService(df_te)
    df_tr, df_te = nslkddFlag(df_tr), nslkddFlag(df_te)
    
    df_tr, df_te = standardize_df_off_tr(df_tr, df_te)
    
    X_tr, y_tr = df_tr.values[:,:-1], df_tr['target'].values
    X_te, y_te = df_te.values[:,:-1], df_te['target'].values

    # column has nans so we are going to get rid of it. 
    X_tr = np.delete(X_tr, 19, 1)
    X_te = np.delete(X_te, 19, 1)

    return X_tr, y_tr, X_te, y_te

def load_awid():
    drop_cols = ['Unnamed: 0']
    df_tr = pd.read_csv('data/AWID/awid_training.csv')
    df_te = pd.read_csv('data/AWID/awid_testing.csv')

    df_tr = df_tr.sample(frac=1).reset_index(drop=True).drop(drop_cols, axis = 1)
    df_te = df_te.sample(frac=1).reset_index(drop=True).drop(drop_cols, axis = 1)
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
            ssd = df_tr[key].values.std()
            if np.abs(ssd) < .0001: 
                ssd = .001
            # scale the testing data w/ the training means/stds
            df_te[key] = (df_te[key].values - df_tr[key].values.mean())/ssd
            # scale the training data 
            df_tr[key] = (df_tr[key].values - df_tr[key].values.mean())/ssd
    return df_tr, df_te


def generate_exploratory_adversarial_data(X_tr:np.ndarray, 
                                          y_tr:np.ndarray, 
                                          X:np.ndarray, 
                                          ctype:str='svc', 
                                          atype:str='fgsm'):
    """Generate adversarial data samples for exploratory attacks 

    This function can generate exploratory adversarial data. Note that some of 
    these datasets take a very long time to run. This is esp the case with the 
    Carlini-Wagner attack. 

    :param X_tr: Dataset 
    :param y_tr: Labels vector 
    :param X: Dataset that is the seed for the adversarial samples
    :param ctype: String that is the classifier used to generate the attack ['svc', 'dt', 'mlp']
    :param atype: String that is the attack type ['fgsm', 'dt', 'deepfool', 'cw', 'pgd']
    :return Numpy array with the adversarial attack dataset
    """
    if ctype == 'svc': 
        # set the classsifier as the svc 
        clfr = SVC(C=1.0, kernel='rbf') 
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
        clfr.fit(X_train, Y_train, epochs=10, batch_size=250, verbose=0, validation_split=0.2)
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


def generate_causative_adversarial_data(X_tr:np.ndarray, 
                                       y_tr:np.ndarray, 
                                       X:np.ndarray, 
                                       y:np.ndarray,
                                       max_iter:int=10,
                                       pp_poison:float=0.33,
                                       atype:str='cleanlabel_pattern'):
    """Generate data for causative adversarial attacks 

    This function can be used to generate causative adversarial attacks. Currently 
    there are three attacks that are implemented. Note that each of them take a 
    long time to run, therefore, it is recommended that you run the code for a while 
    on a machine that is only going to generate the data. Note that you'll get a 
    time out error on Google Colab. 

    :param X_tr: training features 
    :param y_tr: training labels 
    :param X: seeds to train adversarial data 
    :param y: labels to train adversarial data 
    :param max_iter: number of optimization to run for the differnet attacks 
    :param pp_poison: poisoning percents (not really used)
    :param atype: String that is the attack type ['cleanlabel_pattern', 'cleanlabel_single', 'svm']
    """

    if atype == 'cleanlabel_pattern': 
        # run the backdoor cleanlabel pattern attack 
        clfr = SVC(C=1.0, kernel='rbf')
        ytr_ohe = tf.keras.utils.to_categorical(y_tr, 2)
        y_ohe = tf.keras.utils.to_categorical(y, 2)
        clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
        clfr.fit(X_tr, ytr_ohe)
        backdoor = PoisoningAttackBackdoor(add_pattern_bd)  
        # target [1,0] class which is the normal data 
        attack = PoisoningAttackCleanLabelBackdoor(backdoor, 
                                                   clfr, 
                                                   np.array([1,0]), 
                                                   pp_poison=pp_poison, 
                                                   max_iter=max_iter)
        Xadv, yadv = attack.poison(x=np.array(X), y=np.array(y_ohe))
    elif atype == 'cleanlabel_single': 
        # run the backdoor cleanlabel single attack 
        clfr = SVC(C=1.0, kernel='rbf')
        ytr_ohe = tf.keras.utils.to_categorical(y_tr, 2)
        y_ohe = tf.keras.utils.to_categorical(y, 2)
        clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
        clfr.fit(X_tr, ytr_ohe)
        backdoor = PoisoningAttackBackdoor(add_single_bd)
        attack = PoisoningAttackCleanLabelBackdoor(backdoor, 
                                                   clfr, 
                                                   np.array([1,0]), 
                                                   pp_poison=pp_poison, 
                                                   max_iter=max_iter)
        Xadv, yadv = attack.poison(x=np.array(X), y=np.array(y_ohe))
    elif atype == 'svm':
        # run the support vector machine attack 
        n = int(.8*len(y_tr))
        clfr = SVC(C=1.0, kernel='rbf')
        ytr_ohe = tf.keras.utils.to_categorical(y_tr, 2)
        y_ohe = tf.keras.utils.to_categorical(y, 2)
        clfr = SklearnClassifier(clfr, clip_values=(-5.,5.))
        clfr.fit(X_tr[:n].astype(np.float64), ytr_ohe[:n].astype(np.float64))
        attack = PoisoningAttackSVM(clfr, 0.01, 
                                    1.0, 
                                    X_tr[:n].astype(np.float64), 
                                    ytr_ohe[:n].astype(np.float64), 
                                    X_tr[n:].astype(np.float64), 
                                    ytr_ohe[n:].astype(np.float64), 
                                    25)
        # errors were thrown when we were not casting the inputs to the poisoning 
        # function. should not be too much overhead . 
        y_ohe = np.ones(y_ohe.shape) - y_ohe
        X = X.astype(np.float64)
        y_ohe = y_ohe.astype(np.float64)
        y_ohe = np.ones((len(y_ohe), 2)) - y_ohe
        Xadv, yadv = attack.poison(np.array(X), y=np.array(y_ohe))
    else: 
        raise ValueError('An unknown attack was specified.')

    return Xadv, yadv 
