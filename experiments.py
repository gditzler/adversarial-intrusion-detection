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
import numpy as np

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import KFold

from utils import get_performance


# define the functions that are used to updated the dictionary that has the performances of the 
# different methods on different attacks. 
def init_perfs(MODELS:list, ATTACKS:list, PERFS:list):
    """intialize a dictionary of performances  
    """
    all_perfs = {}
    for m in MODELS: 
        for a in ATTACKS: 
            for p in PERFS: 
                all_perfs[''.join([p, '_', m, '_', a])] = 0.0
    return all_perfs

def update_performances(pdict, accs_if, fss_if,  tprs_if,  tnrs_if,  mccs_if,  accs_svm, 
                        fss_svm, tprs_svm, tnrs_svm, mccs_svm, accs_ee, fss_ee, 
                        tprs_ee, tnrs_ee, mccs_ee, accs_lo, fss_lo, tprs_lo, tnrs_lo, 
                        mccs_lo, ATTACK): 
    """update the dictionary of perfromances 
    """
    pdict[''.join(['accs_if_', ATTACK])] += accs_if
    pdict[''.join(['fss_if_', ATTACK])] += fss_if
    pdict[''.join(['tprs_if_', ATTACK])] += tprs_if
    pdict[''.join(['tnrs_if_', ATTACK])] += tnrs_if
    pdict[''.join(['mccs_if_', ATTACK])] += mccs_if
    pdict[''.join(['accs_svm_', ATTACK])] += accs_svm
    pdict[''.join(['fss_svm_', ATTACK])] += fss_svm
    pdict[''.join(['tprs_svm_', ATTACK])] += tprs_svm
    pdict[''.join(['tnrs_svm_', ATTACK])] += tnrs_svm
    pdict[''.join(['mccs_svm_', ATTACK])] += mccs_svm
    pdict[''.join(['accs_ee_', ATTACK])] += accs_ee
    pdict[''.join(['fss_ee_', ATTACK])] += fss_ee
    pdict[''.join(['tprs_ee_', ATTACK])] += tprs_ee
    pdict[''.join(['tnrs_ee_', ATTACK])] += tnrs_ee
    pdict[''.join(['mccs_ee_', ATTACK])] += mccs_ee
    pdict[''.join(['accs_lo_', ATTACK])] += accs_lo
    pdict[''.join(['fss_lo_',  ATTACK])] += fss_lo
    pdict[''.join(['tprs_lo_', ATTACK])] += tprs_lo
    pdict[''.join(['tnrs_lo_', ATTACK])] += tnrs_lo
    pdict[''.join(['mccs_lo_', ATTACK])] += mccs_lo
    return pdict

def scale_dict(all_perfs:dict, MODELS:list, ATTACKS:list, PERFS:list, TRIALS):
    """intialize a dictionary of performances  
    """
    for m in MODELS: 
        for a in ATTACKS: 
            for p in PERFS: 
                all_perfs[''.join([p, '_', m, '_', a])] /= TRIALS
    return all_perfs 



def run_experiment_exploratory(dataset:str='unswnb15', 
                               trials:int=10, 
                               type:str='attacks_all', 
                               verbose:bool=False): 
    """Run the experiment for exploratory attacks against intrusion detection. 

    This function runs the multiple attacks against several different detection algorithms
    which are OneClassSVM, IsolationForest, LocalOutlierFactor and EllipticEnvelope. The 
    performances are ACC, TPR, TNR and MCC. The function returns nothing, but will write 
    files to 'outputs/'

    param: dataset  Dataset [unswnb15, nslkdd]
    param: trials   Number of cross validation runs to perform 
    param: type     Type of experiment to run [attack_all, attack_only]
    param: verbose  Print stuff to the output?
    """

    if verbose: 
        print(''.join(['Dataset: ', dataset]))
    
    # detection algorithm specific parameters 
    support_fraction = .5
    contamination = .05
    degree = 3
    
    MODELS = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['baseline', 'deepfool', 'fgsm', 'pgd', 'dt']
    PERFS = ['accs', 'fss', 'tprs', 'tnrs', 'mccs']
    OUTPUT_FILE = ''.join(['outputs/results_ids_', type, '_', dataset, '.npz'])

    # load the data from the npz files. note that all of the X_tr, X_te, y_tr and y_te are the same 
    # regarless of the file. the difference is in how the Xaml data are generated from a MLPNN. the 
    # labels of y_te are the initial labels of the adversarial data. 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_dt_dt.npz']), allow_pickle=True)
    X_tr, y_tr, X_te, y_te, X_adv_dt = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 
    
    # load the deepfool data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_deepfool.npz']), allow_pickle=True)
    _, _, _, _, X_adv_deepfool = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml']

    # load the fgsm data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_fgsm.npz']), allow_pickle=True)
    _, _, _, _, X_adv_fgsm = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 

    # load the pgd data 
    data = np.load(''.join(['data/', type, '/full_data_', dataset, '_mlp_pgd.npz']), allow_pickle=True)
    _, _, _, _, X_adv_pgd = data['Xtr'], data['ytr'], data['Xte'], data['yte'], data['Xaml'] 

    
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

        if verbose: 
            print(''.join(['   > Running ', str(ell+1), ' of ', str(trials)]))
        ell += 1

        # split the original data into training / testing datasets. we are not going to 
        # use the testing data since we are not going to learn a classifier. 
        X_tr_n, y_tr_n = X_tr[train_index,:], y_tr[train_index]

        # set the normal data 
        X_tr_n_normal = X_tr_n[y_tr_n == 1]

        # isolation forest 
        model = IsolationForest(contamination=contamination).fit(X_tr_n_normal)
        y_if, y_if_deepfool, y_if_fgsm, y_if_pgd, y_if_dt = model.predict(X_te), model.predict(X_adv_deepfool), \
            model.predict(X_adv_fgsm), model.predict(X_adv_pgd), model.predict(X_adv_dt)
        
        # osvm 
        model = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_normal)
        y_svm, y_svm_deepfool, y_svm_fgsm, y_svm_pgd, y_svm_dt = model.predict(X_te), model.predict(X_adv_deepfool), \
            model.predict(X_adv_fgsm), model.predict(X_adv_pgd), model.predict(X_adv_dt)

        # elliiptic 
        model = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_normal)
        y_ee, y_ee_deepfool, y_ee_fgsm, y_ee_pgd, y_ee_dt = model.predict(X_te), model.predict(X_adv_deepfool), \
            model.predict(X_adv_fgsm), model.predict(X_adv_pgd), model.predict(X_adv_dt)

        # local outliers
        model = LocalOutlierFactor(contamination=contamination).fit(X_tr_n_normal)
        # the novelty flag needs to be set to run
        model.novelty = True
        y_lo, y_lo_deepfool, y_lo_fgsm, y_lo_pgd, y_lo_dt = model.predict(X_te), model.predict(X_adv_deepfool), \
            model.predict(X_adv_fgsm), model.predict(X_adv_pgd), model.predict(X_adv_dt)


        # once each of the models have been learned, we need to get the performances then add them to the cumulative performance. 
        # note this nees to be performed for each type of attack. the performances (minus the baseline) will be measured as detection 
        # rates.
        
        # 
        # BASELINE  
        acc_if_baseline, fs_if_baseline, tpr_if_baseline, tnr_if_baseline, mcc_if_baseline = get_performance(y_true=y_te, y_hat=y_if)
        acc_svm_baseline, fs_svm_baseline, tpr_svm_baseline, tnr_svm_baseline, mcc_svm_baseline = get_performance(y_true=y_te, y_hat=y_svm)
        acc_ee_baseline, fs_ee_baseline, tpr_ee_baseline, tnr_ee_baseline, mcc_ee_baseline = get_performance(y_true=y_te, y_hat=y_ee)
        acc_lo_baseline, fs_lo_baseline, tpr_lo_baseline, tnr_lo_baseline, mcc_lo_baseline = get_performance(y_true=y_te, y_hat=y_lo)
        
        all_perfs = update_performances(all_perfs, accs_if=acc_if_baseline, fss_if=fs_if_baseline, tprs_if=tpr_if_baseline,
            tnrs_if=tnr_if_baseline, mccs_if=mcc_if_baseline, accs_svm=acc_svm_baseline, fss_svm=fs_svm_baseline, 
            tprs_svm=tpr_svm_baseline, tnrs_svm=tnr_svm_baseline, mccs_svm=mcc_svm_baseline, accs_ee=acc_ee_baseline, 
            fss_ee=fs_ee_baseline, tprs_ee=tpr_ee_baseline, tnrs_ee=tnr_ee_baseline, mccs_ee=mcc_ee_baseline,
            accs_lo=acc_lo_baseline, fss_lo=fs_lo_baseline, tprs_lo=tpr_lo_baseline, tnrs_lo=tnr_lo_baseline, 
            mccs_lo=mcc_lo_baseline, ATTACK='baseline')
        
        # 
        # DEEPFOOL 
        acc_if_deepfool, fs_if_deepfool, tpr_if_deepfool, tnr_if_deepfool, mcc_if_deepfool = get_performance(y_true=y_aml, y_hat=y_if_deepfool)
        acc_svm_deepfool, fs_svm_deepfool, tpr_svm_deepfool, tnr_svm_deepfool, mcc_svm_deepfool = get_performance(y_true=y_aml, y_hat=y_svm_deepfool)
        acc_ee_deepfool, fs_ee_deepfool, tpr_ee_deepfool, tnr_ee_deepfool, mcc_ee_deepfool = get_performance(y_true=y_aml, y_hat=y_ee_deepfool)
        acc_lo_deepfool, fs_lo_deepfool, tpr_lo_deepfool, tnr_lo_deepfool, mcc_lo_deepfool = get_performance(y_true=y_aml, y_hat=y_lo_deepfool)

        all_perfs = update_performances(all_perfs, accs_if=acc_if_deepfool, fss_if=fs_if_deepfool, tprs_if=tpr_if_deepfool,
            tnrs_if=tnr_if_deepfool, mccs_if=mcc_if_deepfool, accs_svm=acc_svm_deepfool, fss_svm=fs_svm_deepfool, 
            tprs_svm=tpr_svm_deepfool, tnrs_svm=tnr_svm_deepfool, mccs_svm=mcc_svm_deepfool, accs_ee=acc_ee_deepfool, 
            fss_ee=fs_ee_deepfool, tprs_ee=tpr_ee_deepfool, tnrs_ee=tnr_ee_deepfool, mccs_ee=mcc_ee_deepfool,
            accs_lo=acc_lo_deepfool, fss_lo=fs_lo_deepfool, tprs_lo=tpr_lo_deepfool, tnrs_lo=tnr_lo_deepfool, 
            mccs_lo=mcc_lo_deepfool, ATTACK='deepfool')

        # 
        # FGSM
        acc_if_fgsm, fs_if_fgsm, tpr_if_fgsm, tnr_if_fgsm, mcc_if_fgsm = get_performance(y_true=y_aml, y_hat=y_if_fgsm)
        acc_svm_fgsm, fs_svm_fgsm, tpr_svm_fgsm, tnr_svm_fgsm, mcc_svm_fgsm = get_performance(y_true=y_aml, y_hat=y_svm_fgsm)
        acc_ee_fgsm, fs_ee_fgsm, tpr_ee_fgsm, tnr_ee_fgsm, mcc_ee_fgsm = get_performance(y_true=y_aml, y_hat=y_ee_fgsm)
        acc_lo_fgsm, fs_lo_fgsm, tpr_lo_fgsm, tnr_lo_fgsm, mcc_lo_fgsm = get_performance(y_true=y_aml, y_hat=y_lo_fgsm)

        all_perfs = update_performances(all_perfs, accs_if=acc_if_fgsm, fss_if=fs_if_fgsm, tprs_if=tpr_if_fgsm,
            tnrs_if=tnr_if_fgsm, mccs_if=mcc_if_fgsm, accs_svm=acc_svm_fgsm, fss_svm=fs_svm_fgsm, 
            tprs_svm=tpr_svm_fgsm, tnrs_svm=tnr_svm_fgsm, mccs_svm=mcc_svm_fgsm, accs_ee=acc_ee_fgsm, 
            fss_ee=fs_ee_fgsm, tprs_ee=tpr_ee_fgsm, tnrs_ee=tnr_ee_fgsm, mccs_ee=mcc_ee_fgsm,
            accs_lo=acc_lo_fgsm, fss_lo=fs_lo_fgsm, tprs_lo=tpr_lo_fgsm, tnrs_lo=tnr_lo_fgsm, 
            mccs_lo=mcc_lo_fgsm, ATTACK='fgsm')
        
        # 
        # PGD 
        acc_if_pgd, fs_if_pgd, tpr_if_pgd, tnr_if_pgd, mcc_if_pgd = get_performance(y_true=y_aml, y_hat=y_if_pgd)
        acc_svm_pgd, fs_svm_pgd, tpr_svm_pgd, tnr_svm_pgd, mcc_svm_pgd = get_performance(y_true=y_aml, y_hat=y_svm_pgd)
        acc_ee_pgd, fs_ee_pgd, tpr_ee_pgd, tnr_ee_pgd, mcc_ee_pgd = get_performance(y_true=y_aml, y_hat=y_ee_pgd)
        acc_lo_pgd, fs_lo_pgd, tpr_lo_pgd, tnr_lo_pgd, mcc_lo_pgd = get_performance(y_true=y_aml, y_hat=y_lo_pgd)
        
        all_perfs = update_performances(all_perfs, accs_if=acc_if_pgd, fss_if=fs_if_pgd, tprs_if=tpr_if_pgd,
            tnrs_if=tnr_if_pgd, mccs_if=mcc_if_pgd, accs_svm=acc_svm_pgd, fss_svm=fs_svm_pgd, 
            tprs_svm=tpr_svm_pgd, tnrs_svm=tnr_svm_pgd, mccs_svm=mcc_svm_pgd, accs_ee=acc_ee_pgd, 
            fss_ee=fs_ee_pgd, tprs_ee=tpr_ee_pgd, tnrs_ee=tnr_ee_pgd, mccs_ee=mcc_ee_pgd,
            accs_lo=acc_lo_pgd, fss_lo=fs_lo_pgd, tprs_lo=tpr_lo_pgd, tnrs_lo=tnr_lo_pgd, 
            mccs_lo=mcc_lo_pgd, ATTACK='pgd')
        
        # 
        # DECISION TREES 
        acc_if_dt, fs_if_dt, tpr_if_dt, tnr_if_dt, mcc_if_dt = get_performance(y_true=y_aml, y_hat=y_if_dt)
        acc_svm_dt, fs_svm_dt, tpr_svm_dt, tnr_svm_dt, mcc_svm_dt = get_performance(y_true=y_aml, y_hat=y_svm_dt)
        acc_ee_dt, fs_ee_dt, tpr_ee_dt, tnr_ee_dt, mcc_ee_dt = get_performance(y_true=y_aml, y_hat=y_ee_dt)
        acc_lo_dt, fs_lo_dt, tpr_lo_dt, tnr_lo_dt, mcc_lo_dt = get_performance(y_true=y_aml, y_hat=y_lo_dt)
        
        all_perfs = update_performances(all_perfs, accs_if=acc_if_dt, fss_if=fs_if_dt, tprs_if=tpr_if_dt,
            tnrs_if=tnr_if_dt, mccs_if=mcc_if_dt, accs_svm=acc_svm_dt, fss_svm=fs_svm_dt, 
            tprs_svm=tpr_svm_dt, tnrs_svm=tnr_svm_dt, mccs_svm=mcc_svm_dt, accs_ee=acc_ee_dt, 
            fss_ee=fs_ee_dt, tprs_ee=tpr_ee_dt, tnrs_ee=tnr_ee_dt, mccs_ee=mcc_ee_dt,
            accs_lo=acc_lo_dt, fss_lo=fs_lo_dt, tprs_lo=tpr_lo_dt, tnrs_lo=tnr_lo_dt, 
            mccs_lo=mcc_lo_dt, ATTACK='dt')
        
    # scale by the number of trials that we run.
    all_perfs = scale_dict(all_perfs, MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS, TRIALS=trials)
    
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')

    np.savez(OUTPUT_FILE, all_pers = all_perfs)

    return None 


def run_experiment_causative(dataset:str='nslkdd', 
                             trials:int=10, 
                             ppoison:float=0.1, 
                             verbose:bool=False): 
    """run the causative experiments 
    """
    
    if verbose: 
        print(''.join(['Dataset: ', dataset]))
    
    # detection algorithm specific parameters 
    support_fraction = .5
    contamination = .05
    degree = 3
    
    MODELS = ['if', 'svm', 'ee', 'lo']
    ATTACKS = ['', 'pattern', 'single', 'svc']
    PERFS = ['accs', 'fss', 'tprs', 'tnrs', 'mccs']
    OUTPUT_FILE = ''.join(['outputs/results_ids_causative_', dataset,'_pp', str(int(100*ppoison)), '.npz'])

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
        Xa = Xa[np.random.randint(0, len(Xa), m)]
        X_tr_n_pattern = np.concatenate((X_tr_n_normal, Xa), axis=0)
        
        Xa = X_adv_single[y_adv_single==0]
        Xa = Xa[np.random.randint(0, len(Xa), m)]
        X_tr_n_single = np.concatenate((X_tr_n_normal, Xa), axis=0)
        
        Xa = X_adv_svc[y_adv_svc==0]
        Xa = Xa[np.random.randint(0, len(Xa), m)]
        X_tr_n_svc = np.concatenate((X_tr_n_normal, Xa), axis=0)

        # isolation forest 
        model_n = IsolationForest(contamination=contamination).fit(X_tr_n_normal)
        model_a_pattern = IsolationForest(contamination=contamination).fit(X_tr_n_pattern)
        model_a_single = IsolationForest(contamination=contamination).fit(X_tr_n_single)
        model_a_svc = IsolationForest(contamination=contamination).fit(X_tr_n_svc)
        y_if, y_if_pattern, y_if_single, y_if_svc = model_n.predict(X_te), model_a_pattern.predict(X_te), \
            model_a_single.predict(X_te), model_a_svc.predict(X_te)
        
        # osvm 
        model_n = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_normal)
        model_a_pattern = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_pattern)
        model_a_single = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_single)
        model_a_svc = OneClassSVM(kernel='poly', degree=degree).fit(X_tr_n_svc) 
        y_svm, y_svm_pattern, y_svm_single, y_svm_svc = model_n.predict(X_te), model_a_pattern.predict(X_te), \
            model_a_single.predict(X_te), model_a_svc.predict(X_te)

        # elliiptic 
        model_n = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_normal)
        model_a_pattern = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_pattern)
        model_a_single = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_single)
        model_a_svc = EllipticEnvelope(contamination=contamination, support_fraction=support_fraction).fit(X_tr_n_svc) 
        
        y_ee, y_ee_pattern, y_ee_single, y_ee_svc= model_n.predict(X_te), model_a_pattern.predict(X_te), \
            model_a_single.predict(X_te), model_a_svc.predict(X_te)

        # local outliers
        model_n = LocalOutlierFactor(contamination=contamination).fit(X_tr_n_normal)
        model_a_pattern = LocalOutlierFactor(contamination=contamination).fit(X_tr_n_pattern)
        model_a_single = LocalOutlierFactor(contamination=contamination).fit(X_tr_n_single)
        model_a_svc = LocalOutlierFactor(contamination=contamination).fit(X_tr_n_svc)
        
        # the novelty flag needs to be set to run
        model_n.novelty = True
        model_a_svc.novelty = True
        model_a_pattern.novelty = True
        model_a_single.novelty = True
        y_lo, y_lo_pattern, y_lo_single, y_lo_svc = model_n.predict(X_te), model_a_pattern.predict(X_te), \
            model_a_single.predict(X_te), model_a_svc.predict(X_te)

        # get the prediction rates 
        acc_if_baseline, fs_if_baseline, tpr_if_baseline, tnr_if_baseline, mcc_if_baseline = get_performance(y_true=y_te, y_hat=y_if)
        acc_if_pattern, fs_if_pattern, tpr_if_pattern, tnr_if_pattern, mcc_if_pattern = get_performance(y_true=y_te, y_hat=y_if_pattern)
        acc_if_single, fs_if_single, tpr_if_single, tnr_if_single, mcc_if_single = get_performance(y_true=y_te, y_hat=y_if_single)
        acc_if_svc, fs_if_svc, tpr_if_svc, tnr_if_svc, mcc_if_svc = get_performance(y_true=y_te, y_hat=y_if_svc)

        acc_svm_baseline, fs_svm_baseline, tpr_svm_baseline, tnr_svm_baseline, mcc_svm_baseline = get_performance(y_true=y_te, y_hat=y_svm)
        acc_svm_pattern, fs_svm_pattern, tpr_svm_pattern, tnr_svm_pattern, mcc_svm_pattern = get_performance(y_true=y_te, y_hat=y_svm_pattern)
        acc_svm_single, fs_svm_single, tpr_svm_single, tnr_svm_single, mcc_svm_single = get_performance(y_true=y_te, y_hat=y_svm_single)
        acc_svm_svc, fs_svm_svc, tpr_svm_svc, tnr_svm_svc, mcc_svm_svc = get_performance(y_true=y_te, y_hat=y_svm_svc)
        
        acc_ee_baseline, fs_ee_baseline, tpr_ee_baseline, tnr_ee_baseline, mcc_ee_baseline = get_performance(y_true=y_te, y_hat=y_ee)
        acc_ee_pattern, fs_ee_pattern, tpr_ee_pattern, tnr_ee_pattern, mcc_ee_pattern = get_performance(y_true=y_te, y_hat=y_ee_pattern)
        acc_ee_single, fs_ee_single, tpr_ee_single, tnr_ee_single, mcc_ee_single = get_performance(y_true=y_te, y_hat=y_ee_single)
        acc_ee_svc, fs_ee_svc, tpr_ee_svc, tnr_ee_svc, mcc_ee_svc = get_performance(y_true=y_te, y_hat=y_ee_svc)
        
        acc_lo_baseline, fs_lo_baseline, tpr_lo_baseline, tnr_lo_baseline, mcc_lo_baseline = get_performance(y_true=y_te, y_hat=y_lo)
        acc_lo_pattern, fs_lo_pattern, tpr_lo_pattern, tnr_lo_pattern, mcc_lo_pattern = get_performance(y_true=y_te, y_hat=y_lo_pattern)
        acc_lo_single, fs_lo_single, tpr_lo_single, tnr_lo_single, mcc_lo_single = get_performance(y_true=y_te, y_hat=y_lo_single)
        acc_lo_svc, fs_lo_svc, tpr_lo_svc, tnr_lo_svc, mcc_lo_svc = get_performance(y_true=y_te, y_hat=y_lo_svc)
        
        
        # SAVE THE PERFORMANCES 
        all_perfs = update_performances(all_perfs, accs_if=acc_if_baseline, fss_if=fs_if_baseline, tprs_if=tpr_if_baseline,
            tnrs_if=tnr_if_baseline, mccs_if=mcc_if_baseline, accs_svm=acc_svm_baseline, fss_svm=fs_svm_baseline, 
            tprs_svm=tpr_svm_baseline, tnrs_svm=tnr_svm_baseline, mccs_svm=mcc_svm_baseline, accs_ee=acc_ee_baseline, 
            fss_ee=fs_ee_baseline, tprs_ee=tpr_ee_baseline, tnrs_ee=tnr_ee_baseline, mccs_ee=mcc_ee_baseline,
            accs_lo=acc_lo_baseline, fss_lo=fs_lo_baseline, tprs_lo=tpr_lo_baseline, tnrs_lo=tnr_lo_baseline, 
            mccs_lo=mcc_lo_baseline, ATTACK='baseline')

        all_perfs = update_performances(all_perfs, accs_if=acc_if_pattern, fss_if=fs_if_pattern, tprs_if=tpr_if_pattern,
            tnrs_if=tnr_if_pattern, mccs_if=mcc_if_pattern, accs_svm=acc_svm_pattern, fss_svm=fs_svm_pattern, 
            tprs_svm=tpr_svm_pattern, tnrs_svm=tnr_svm_pattern, mccs_svm=mcc_svm_pattern, accs_ee=acc_ee_pattern, 
            fss_ee=fs_ee_pattern, tprs_ee=tpr_ee_pattern, tnrs_ee=tnr_ee_pattern, mccs_ee=mcc_ee_pattern,
            accs_lo=acc_lo_pattern, fss_lo=fs_lo_pattern, tprs_lo=tpr_lo_pattern, tnrs_lo=tnr_lo_pattern, 
            mccs_lo=mcc_lo_pattern, ATTACK='pattern')
        
        all_perfs = update_performances(all_perfs, accs_if=acc_if_single, fss_if=fs_if_single, tprs_if=tpr_if_single,
            tnrs_if=tnr_if_single, mccs_if=mcc_if_single, accs_svm=acc_svm_single, fss_svm=fs_svm_single, 
            tprs_svm=tpr_svm_single, tnrs_svm=tnr_svm_single, mccs_svm=mcc_svm_single, accs_ee=acc_ee_single, 
            fss_ee=fs_ee_single, tprs_ee=tpr_ee_single, tnrs_ee=tnr_ee_single, mccs_ee=mcc_ee_single,
            accs_lo=acc_lo_single, fss_lo=fs_lo_single, tprs_lo=tpr_lo_single, tnrs_lo=tnr_lo_single, 
            mccs_lo=mcc_lo_single, ATTACK='single')

        all_perfs = update_performances(all_perfs, accs_if=acc_if_svc, fss_if=fs_if_svc, tprs_if=tpr_if_svc,
            tnrs_if=tnr_if_svc, mccs_if=mcc_if_svc, accs_svm=acc_svm_svc, fss_svm=fs_svm_svc, 
            tprs_svm=tpr_svm_svc, tnrs_svm=tnr_svm_svc, mccs_svm=mcc_svm_svc, accs_ee=acc_ee_svc, 
            fss_ee=fs_ee_svc, tprs_ee=tpr_ee_svc, tnrs_ee=tnr_ee_svc, mccs_ee=mcc_ee_svc,
            accs_lo=acc_lo_svc, fss_lo=fs_lo_svc, tprs_lo=tpr_lo_svc, tnrs_lo=tnr_lo_svc, 
            mccs_lo=mcc_lo_svc, ATTACK='svc')


    # scale by the number of trials
    all_perfs = scale_dict(all_perfs, MODELS=MODELS, ATTACKS=ATTACKS, PERFS=PERFS, TRIALS=trials)
    
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')

    np.savez(OUTPUT_FILE, all_perfs = all_perfs)
    
    return None 
