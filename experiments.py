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

import os 
import tqdm 
import pickle 
import numpy as np

from sklearn.model_selection import KFold
from models import evaluate_models_exploratory, evaluate_models_causative
from utils import init_perfs, scale_dict



def run_experiment_exploratory(dataset:str='unswnb15', 
                               trials:int=10, 
                               type:str='attacks_all'): 
    """Run the experiment for exploratory attacks against intrusion detection. 

    This function runs the multiple exploratory attacks against several detection algorithms
    which are OneClassSVM, IsolationForest, LocalOutlierFactor and EllipticEnvelope. The 
    performances are ACC, TPR, TNR and MCC. The function returns nothing, but will write 
    files to 'outputs/'

    param: dataset  Dataset [nslkdd]
    param: trials   Number of cross validation runs to perform 
    param: type     Type of experiment to run [attack_all, attack_only]
    """

    print(''.join(['Dataset: ', dataset, ' (exploratory)']))
    
    # detection algorithm specific parameters 
    support_fraction = .5
    contamination = .05
    degree = 3
    
    MODELS = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['baseline', 'deepfool', 'fgsm', 'pgd', 'dt']
    PERFS = ['accs', 'fss', 'tprs', 'tnrs', 'mccs']
    OUTPUT_FILE = ''.join(['outputs/results_ids_', type, '_', dataset, '.pkl'])

    # load the data from the npz files. note that all of the X_tr, X_te, y_tr and y_te are the same 
    # regarless of the file. the difference is in how the Xaml data are generated from a MLPNN. the 
    # labels of y_te are the initial labels of the adversarial data. 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_dt_dt.npz']), allow_pickle=True)
    X_tr, y_tr, X_te, y_te, X_adv_dt = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 
    
    # load the deepfool data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_deepfool.npz']), allow_pickle=True)
    X_adv_deepfool = data['Xaml']

    # load the fgsm data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_fgsm.npz']), allow_pickle=True)
    X_adv_fgsm = data['Xaml'] 

    # load the pgd data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_pgd.npz']), allow_pickle=True)
    X_adv_pgd = data['Xaml'] 

    
    # there are two types of experimenta that we can run. first, we need to set the class labels into ones 
    # that we can work with in this code. when we load the data the labels are 0 and 1, but the rest of the 
    # code needs to have them set to +/-1. here is the description of attacks_all and attacks_one: 
    #  - attacks_all: the labels for the adversarial data are the class labels of the original data. this 
    #    is essentially labeling the data with the original labels and seeing if the model can still classify 
    #    the image as ``out of sample'' 
    #  - attacks_only: the labels for all of the adversarial data are set to -1. we generated the data using
    #    since the attack data were only generate on malicous data in the original set. that is the attack 
    #    data are used to generate adversarial data. so from the adversaries POV they are taking malicious 
    #    data that they want the user to classify as normal 
    if type == 'attacks_all': 
        # change the labels; 1=normal; -1=maliicious
        y_tr[y_tr==1] = -1
        y_tr[y_tr==0] = 1
        y_te[y_te==1] = -1
        y_te[y_te==0] =  1
        y_aml = y_te
    elif type == 'attacks_only': 
        # change the labels; 1=normal; -1=maliicious
        y_tr[y_tr==1] = -1
        y_tr[y_tr==0] = 1
        y_te[y_te==1] = -1
        y_te[y_te==0] =  1
        y_aml = -np.ones(len(X_adv_pgd))
    else: 
        raise ValueError('Unknown type: attacks_only or attacks_all')


    # we need to set up the k-fold evaluator 
    kf = KFold(n_splits=trials)
    
    all_perfs = init_perfs(MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS) 

    ell = 0
    for train_index, _ in kf.split(X_tr):

        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        X_tr_n, y_tr_n = X_tr[train_index,:], y_tr[train_index]

        # set the normal data 
        X_tr_n_normal = X_tr_n[y_tr_n == 1]

        all_perfs = evaluate_models_exploratory(X_normal=X_tr_n_normal, 
                                                X_te=X_te, 
                                                X_adv_deepfool=X_adv_deepfool, 
                                                X_adv_fgsm=X_adv_fgsm, 
                                                X_adv_pgd=X_adv_pgd, 
                                                X_adv_dt=X_adv_dt, 
                                                Y=y_te,
                                                Y_aml=y_aml, 
                                                perfs=all_perfs, 
                                                contamination=contamination, 
                                                degree=degree, 
                                                support_fraction=support_fraction)

    # scale by the number of trials that we run.
    all_perfs = scale_dict(all_perfs, MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS, TRIALS=trials)
    
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')

    pickle.dump(all_perfs, open(OUTPUT_FILE, 'wb'))


def run_experiment_causative(dataset:str='nslkdd', 
                             trials:int=10, 
                             ppoison:float=0.1): 
    """run the causative experiments

    This function runs the multiple causative attacks against several detection algorithms
    which are OneClassSVM, IsolationForest, LocalOutlierFactor and EllipticEnvelope. The 
    performances are ACC, TPR, TNR and MCC. The function returns nothing, but will write 
    files to 'outputs/'

    param: dataset  Dataset [nslkdd]
    param: trials   Number of cross validation runs to perform 
    param: type     Type of experiment to run [attack_all, attack_only]
    """

    print(''.join(['Dataset: ', dataset, ' (causative)']))
    
    # detection algorithm specific parameters 
    support_fraction = .5
    contamination = .05
    degree = 3
    
    MODELS = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['baseline', 'pattern', 'single', 'svc']
    PERFS = ['accs', 'fss', 'tprs', 'tnrs', 'mccs']
    OUTPUT_FILE = ''.join(['outputs/results_ids_causative_', dataset,'_pp', str(int(100*ppoison)), '.pkl'])

    # load the data from the npz files. note that all of the X_tr, X_te, y_tr and y_te are the same 
    # regarless of the file. the difference is in how the Xaml data are generated from a MLPNN. the 
    # labels of y_te are the initial labels of the adversarial data. 
    data = np.load(''.join(['data/causative/full_data_', dataset, '_cleanlabel_pattern.npz']), allow_pickle=True)
    X_tr, y_tr, X_te, y_te, X_adv_pattern, y_adv_pattern  = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'], np.argmax(data['yaml'], axis=1) 
    
    data = np.load(''.join(['data/causative/full_data_', dataset, '_cleanlabel_single.npz']), allow_pickle=True)
    X_adv_single, y_adv_single = data['Xaml'], np.argmax(data['yaml'], axis=1) 
    
    data = np.load(''.join(['data/causative/full_data_', dataset, '_svc.npz']), allow_pickle=True)
    X_adv_svc, y_adv_svc = data['Xaml'], np.argmax(data['yaml'], axis=1) 

    all_perfs = init_perfs(MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS)

    # change the labels; 1=normal; -1=maliicious
    y_tr[y_tr==1] = -1
    y_tr[y_tr==0] = 1
    y_te[y_te==1] = -1
    y_te[y_te==0] =  1

    for t in range(trials):
        if verbose: 
            print(''.join(['   > Running ', str(t+1), ' of ', str(trials)]))

        train_index = np.random.randint(0, len(y_tr), len(y_tr))

        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        X_tr_n, y_tr_n = X_tr[train_index,:], y_tr[train_index]
        
        # set the normal data 
        X_tr_n_normal = X_tr_n[y_tr_n == 1]
        
        # split out the adversarial data 
        m = int(len(X_tr_n_normal)/(1 - ppoison)) - len(X_tr_n_normal)

        Xa = X_adv_pattern[y_adv_pattern==0]
        X_tr_n_pattern = np.concatenate((X_tr_n_normal, Xa[np.random.randint(0, len(Xa), m)]), axis=0)
        
        Xa = X_adv_single[y_adv_single==0]
        X_tr_n_single = np.concatenate((X_tr_n_normal, Xa[np.random.randint(0, len(Xa), m)]), axis=0)
        
        Xa = X_adv_svc[y_adv_svc==0]
        X_tr_n_svc = np.concatenate((X_tr_n_normal, Xa[np.random.randint(0, len(Xa), m)]), axis=0)

        all_perfs = evaluate_models_causative(X=X_tr_n_normal, 
                                              X_pattern=X_tr_n_pattern, 
                                              X_single=X_tr_n_single, 
                                              X_svc=X_tr_n_svc,
                                              X_te=X_te, 
                                              Y_te=y_te,  
                                              perfs=all_perfs, 
                                              contamination=contamination, 
                                              degree=degree, 
                                              support_fraction=support_fraction)
     
    # scale by the number of trials
    all_perfs = scale_dict(all_perfs, MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS, TRIALS=trials)
    
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')

    pickle.dump(all_perfs, open(OUTPUT_FILE, 'wb'))
