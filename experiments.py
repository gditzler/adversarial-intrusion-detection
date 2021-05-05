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

    # need to intialize the outputs to zeros. not the most efficient way of doing this. 
    accs_if_baseline, fss_if_baseline, tprs_if_baseline, tnrs_if_baseline, mccs_if_baseline = 0., 0., 0., 0., 0.
    accs_svm_baseline, fss_svm_baseline, tprs_svm_baseline, tnrs_svm_baseline, mccs_svm_baseline = 0., 0., 0., 0., 0. 
    accs_ee_baseline, fss_ee_baseline, tprs_ee_baseline, tnrs_ee_baseline, mccs_ee_baseline = 0., 0., 0., 0., 0.
    accs_lo_baseline, fss_lo_baseline, tprs_lo_baseline, tnrs_lo_baseline, mccs_lo_baseline = 0., 0., 0., 0., 0.

    accs_if_deepfool, fss_if_deepfool, tprs_if_deepfool, tnrs_if_deepfool, mccs_if_deepfool = 0., 0., 0., 0., 0.
    accs_svm_deepfool, fss_svm_deepfool, tprs_svm_deepfool, tnrs_svm_deepfool, mccs_svm_deepfool = 0., 0., 0., 0., 0. 
    accs_ee_deepfool, fss_ee_deepfool, tprs_ee_deepfool, tnrs_ee_deepfool, mccs_ee_deepfool = 0., 0., 0., 0., 0.
    accs_lo_deepfool, fss_lo_deepfool, tprs_lo_deepfool, tnrs_lo_deepfool, mccs_lo_deepfool = 0., 0., 0., 0., 0.

    accs_if_fgsm, fss_if_fgsm, tprs_if_fgsm, tnrs_if_fgsm, mccs_if_fgsm = 0., 0., 0., 0., 0.
    accs_svm_fgsm, fss_svm_fgsm, tprs_svm_fgsm, tnrs_svm_fgsm, mccs_svm_fgsm = 0., 0., 0., 0., 0. 
    accs_ee_fgsm, fss_ee_fgsm, tprs_ee_fgsm, tnrs_ee_fgsm, mccs_ee_fgsm = 0., 0., 0., 0., 0.
    accs_lo_fgsm, fss_lo_fgsm, tprs_lo_fgsm, tnrs_lo_fgsm, mccs_lo_fgsm = 0., 0., 0., 0., 0.

    accs_if_pgd, fss_if_pgd, tprs_if_pgd, tnrs_if_pgd, mccs_if_pgd = 0., 0., 0., 0., 0.
    accs_svm_pgd, fss_svm_pgd, tprs_svm_pgd, tnrs_svm_pgd, mccs_svm_pgd = 0., 0., 0., 0., 0. 
    accs_ee_pgd, fss_ee_pgd, tprs_ee_pgd, tnrs_ee_pgd, mccs_ee_pgd = 0., 0., 0., 0., 0.
    accs_lo_pgd, fss_lo_pgd, tprs_lo_pgd, tnrs_lo_pgd, mccs_lo_pgd = 0., 0., 0., 0., 0.

    accs_if_dt, fss_if_dt, tprs_if_dt, tnrs_if_dt, mccs_if_dt = 0., 0., 0., 0., 0.
    accs_svm_dt, fss_svm_dt, tprs_svm_dt, tnrs_svm_dt, mccs_svm_dt = 0., 0., 0., 0., 0. 
    accs_ee_dt, fss_ee_dt, tprs_ee_dt, tnrs_ee_dt, mccs_ee_dt = 0., 0., 0., 0., 0.
    accs_lo_dt, fss_lo_dt, tprs_lo_dt, tnrs_lo_dt, mccs_lo_dt = 0., 0., 0., 0., 0.

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
        acc_if_baseline, fs_if_baseline, tpr_if_baseline, tnr_if_baseline, mcc_if_baseline = get_performance(y_true=y_te, y_hat=y_if)
        acc_svm_baseline, fs_svm_baseline, tpr_svm_baseline, tnr_svm_baseline, mcc_svm_baseline = get_performance(y_true=y_te, y_hat=y_svm)
        acc_ee_baseline, fs_ee_baseline, tpr_ee_baseline, tnr_ee_baseline, mcc_ee_baseline = get_performance(y_true=y_te, y_hat=y_ee)
        acc_lo_baseline, fs_lo_baseline, tpr_lo_baseline, tnr_lo_baseline, mcc_lo_baseline = get_performance(y_true=y_te, y_hat=y_lo)
        
        accs_if_baseline += acc_if_baseline 
        fss_if_baseline += fs_if_baseline 
        tprs_if_baseline += tpr_if_baseline
        tnrs_if_baseline += tnr_if_baseline
        mccs_if_baseline += mcc_if_baseline 
        accs_svm_baseline += acc_svm_baseline
        fss_svm_baseline  += fs_svm_baseline 
        tprs_svm_baseline += tpr_svm_baseline 
        tnrs_svm_baseline += tnr_svm_baseline
        mccs_svm_baseline += mcc_svm_baseline
        accs_ee_baseline += acc_ee_baseline 
        fss_ee_baseline += fs_ee_baseline 
        tprs_ee_baseline += tpr_ee_baseline
        tnrs_ee_baseline += tnr_ee_baseline 
        mccs_ee_baseline += mcc_ee_baseline
        accs_lo_baseline += acc_lo_baseline 
        fss_lo_baseline += fs_lo_baseline 
        tprs_lo_baseline += tpr_lo_baseline 
        tnrs_lo_baseline += tnr_lo_baseline
        mccs_lo_baseline += mcc_lo_baseline
        
        acc_if_deepfool, fs_if_deepfool, tpr_if_deepfool, tnr_if_deepfool, mcc_if_deepfool = get_performance(y_true=y_aml, y_hat=y_if_deepfool)
        acc_svm_deepfool, fs_svm_deepfool, tpr_svm_deepfool, tnr_svm_deepfool, mcc_svm_deepfool = get_performance(y_true=y_aml, y_hat=y_svm_deepfool)
        acc_ee_deepfool, fs_ee_deepfool, tpr_ee_deepfool, tnr_ee_deepfool, mcc_ee_deepfool = get_performance(y_true=y_aml, y_hat=y_ee_deepfool)
        acc_lo_deepfool, fs_lo_deepfool, tpr_lo_deepfool, tnr_lo_deepfool, mcc_lo_deepfool = get_performance(y_true=y_aml, y_hat=y_lo_deepfool)

        accs_if_deepfool += acc_if_deepfool 
        fss_if_deepfool += fs_if_deepfool 
        tprs_if_deepfool += tpr_if_deepfool
        tnrs_if_deepfool += tnr_if_deepfool
        mccs_if_deepfool += mcc_if_deepfool 
        accs_svm_deepfool += acc_svm_deepfool
        fss_svm_deepfool += fs_svm_deepfool 
        tprs_svm_deepfool += tpr_svm_deepfool 
        tnrs_svm_deepfool += tnr_svm_deepfool
        mccs_svm_deepfool += mcc_svm_deepfool
        accs_ee_deepfool += acc_ee_deepfool 
        fss_ee_deepfool += fs_ee_deepfool 
        tprs_ee_deepfool += tpr_ee_deepfool
        tnrs_ee_deepfool += tnr_ee_deepfool 
        mccs_ee_deepfool += mcc_ee_deepfool
        accs_lo_deepfool += acc_lo_deepfool 
        fss_lo_deepfool += fs_lo_deepfool 
        tprs_lo_deepfool += tpr_lo_deepfool 
        tnrs_lo_deepfool += tnr_lo_deepfool
        mccs_lo_deepfool += mcc_lo_deepfool


        acc_if_fgsm, fs_if_fgsm, tpr_if_fgsm, tnr_if_fgsm, mcc_if_fgsm = get_performance(y_true=y_aml, y_hat=y_if_fgsm)
        acc_svm_fgsm, fs_svm_fgsm, tpr_svm_fgsm, tnr_svm_fgsm, mcc_svm_fgsm = get_performance(y_true=y_aml, y_hat=y_svm_fgsm)
        acc_ee_fgsm, fs_ee_fgsm, tpr_ee_fgsm, tnr_ee_fgsm, mcc_ee_fgsm = get_performance(y_true=y_aml, y_hat=y_ee_fgsm)
        acc_lo_fgsm, fs_lo_fgsm, tpr_lo_fgsm, tnr_lo_fgsm, mcc_lo_fgsm = get_performance(y_true=y_aml, y_hat=y_lo_fgsm)

        accs_if_fgsm += acc_if_fgsm 
        fss_if_fgsm += fs_if_fgsm 
        tprs_if_fgsm += tpr_if_fgsm
        tnrs_if_fgsm += tnr_if_fgsm
        mccs_if_fgsm += mcc_if_fgsm 
        accs_svm_fgsm += acc_svm_fgsm
        fss_svm_fgsm += fs_svm_fgsm 
        tprs_svm_fgsm += tpr_svm_fgsm 
        tnrs_svm_fgsm += tnr_svm_fgsm
        mccs_svm_fgsm += mcc_svm_fgsm
        accs_ee_fgsm += acc_ee_fgsm 
        fss_ee_fgsm += fs_ee_fgsm 
        tprs_ee_fgsm += tpr_ee_fgsm
        tnrs_ee_fgsm += tnr_ee_fgsm 
        mccs_ee_fgsm += mcc_ee_fgsm
        accs_lo_fgsm += acc_lo_fgsm 
        fss_lo_fgsm += fs_lo_fgsm 
        tprs_lo_fgsm += tpr_lo_fgsm 
        tnrs_lo_fgsm += tnr_lo_fgsm
        mccs_lo_fgsm += mcc_lo_fgsm


        acc_if_pgd, fs_if_pgd, tpr_if_pgd, tnr_if_pgd, mcc_if_pgd = get_performance(y_true=y_aml, y_hat=y_if_pgd)
        acc_svm_pgd, fs_svm_pgd, tpr_svm_pgd, tnr_svm_pgd, mcc_svm_pgd = get_performance(y_true=y_aml, y_hat=y_svm_pgd)
        acc_ee_pgd, fs_ee_pgd, tpr_ee_pgd, tnr_ee_pgd, mcc_ee_pgd = get_performance(y_true=y_aml, y_hat=y_ee_pgd)
        acc_lo_pgd, fs_lo_pgd, tpr_lo_pgd, tnr_lo_pgd, mcc_lo_pgd = get_performance(y_true=y_aml, y_hat=y_lo_pgd)

        accs_if_pgd += acc_if_pgd 
        fss_if_pgd += fs_if_pgd 
        tprs_if_pgd += tpr_if_pgd
        tnrs_if_pgd += tnr_if_pgd
        mccs_if_pgd += mcc_if_pgd 
        accs_svm_pgd += acc_svm_pgd
        fss_svm_pgd += fs_svm_pgd 
        tprs_svm_pgd += tpr_svm_pgd 
        tnrs_svm_pgd += tnr_svm_pgd
        mccs_svm_pgd += mcc_svm_pgd
        accs_ee_pgd += acc_ee_pgd 
        fss_ee_pgd += fs_ee_pgd 
        tprs_ee_pgd += tpr_ee_pgd
        tnrs_ee_pgd += tnr_ee_pgd 
        mccs_ee_pgd += mcc_ee_pgd
        accs_lo_pgd += acc_lo_pgd 
        fss_lo_pgd += fs_lo_pgd 
        tprs_lo_pgd += tpr_lo_pgd 
        tnrs_lo_pgd += tnr_lo_pgd
        mccs_lo_pgd += mcc_lo_pgd

        acc_if_dt, fs_if_dt, tpr_if_dt, tnr_if_dt, mcc_if_dt = get_performance(y_true=y_aml, y_hat=y_if_dt)
        acc_svm_dt, fs_svm_dt, tpr_svm_dt, tnr_svm_dt, mcc_svm_dt = get_performance(y_true=y_aml, y_hat=y_svm_dt)
        acc_ee_dt, fs_ee_dt, tpr_ee_dt, tnr_ee_dt, mcc_ee_dt = get_performance(y_true=y_aml, y_hat=y_ee_dt)
        acc_lo_dt, fs_lo_dt, tpr_lo_dt, tnr_lo_dt, mcc_lo_dt = get_performance(y_true=y_aml, y_hat=y_lo_dt)

        accs_if_dt += acc_if_dt 
        fss_if_dt += fs_if_dt 
        tprs_if_dt += tpr_if_dt
        tnrs_if_dt += tnr_if_dt
        mccs_if_dt += mcc_if_dt 
        accs_svm_dt += acc_svm_dt
        fss_svm_dt += fs_svm_dt 
        tprs_svm_dt += tpr_svm_dt 
        tnrs_svm_dt += tnr_svm_dt
        mccs_svm_dt += mcc_svm_dt
        accs_ee_dt += acc_ee_dt 
        fss_ee_dt += fs_ee_dt 
        tprs_ee_dt += tpr_ee_dt
        tnrs_ee_dt += tnr_ee_dt 
        mccs_ee_dt += mcc_ee_dt
        accs_lo_dt += acc_lo_dt 
        fss_lo_dt += fs_lo_dt 
        tprs_lo_dt += tpr_lo_dt 
        tnrs_lo_dt += tnr_lo_dt
        mccs_lo_dt += mcc_lo_dt

    # scale by the number of trials that we run. 
    accs_if_baseline /= trials 
    fss_if_baseline /= trials
    tprs_if_baseline /= trials
    tnrs_if_baseline /= trials
    mccs_if_baseline /= trials
    accs_svm_baseline /= trials
    fss_svm_baseline  /= trials
    tprs_svm_baseline /= trials
    tnrs_svm_baseline /= trials
    mccs_svm_baseline /= trials
    accs_ee_baseline  /= trials
    fss_ee_baseline  /= trials
    tprs_ee_baseline /= trials
    tnrs_ee_baseline /= trials
    mccs_ee_baseline /= trials
    accs_lo_baseline /= trials
    fss_lo_baseline /= trials
    tprs_lo_baseline /= trials
    tnrs_lo_baseline /= trials
    mccs_lo_baseline /= trials


    accs_if_deepfool /= trials 
    fss_if_deepfool /= trials
    tprs_if_deepfool /= trials
    tnrs_if_deepfool /= trials
    mccs_if_deepfool /= trials
    accs_svm_deepfool /= trials
    fss_svm_deepfool  /= trials
    tprs_svm_deepfool /= trials
    tnrs_svm_deepfool /= trials
    mccs_svm_deepfool /= trials
    accs_ee_deepfool  /= trials
    fss_ee_deepfool  /= trials
    tprs_ee_deepfool /= trials
    tnrs_ee_deepfool /= trials
    mccs_ee_deepfool /= trials
    accs_lo_deepfool /= trials
    fss_lo_deepfool /= trials
    tprs_lo_deepfool /= trials
    tnrs_lo_deepfool /= trials
    mccs_lo_deepfool /= trials 

    accs_if_dt /= trials 
    fss_if_dt /= trials
    tprs_if_dt /= trials
    tnrs_if_dt /= trials
    mccs_if_dt /= trials
    accs_svm_dt /= trials
    fss_svm_dt  /= trials
    tprs_svm_dt /= trials
    tnrs_svm_dt /= trials
    mccs_svm_dt /= trials
    accs_ee_dt  /= trials
    fss_ee_dt  /= trials
    tprs_ee_dt /= trials
    tnrs_ee_dt /= trials
    mccs_ee_dt /= trials
    accs_lo_dt /= trials
    fss_lo_dt /= trials
    tprs_lo_dt /= trials
    tnrs_lo_dt /= trials
    mccs_lo_dt /= trials

    accs_if_fgsm /= trials 
    fss_if_fgsm /= trials
    tprs_if_fgsm /= trials
    tnrs_if_fgsm /= trials
    mccs_if_fgsm /= trials
    accs_svm_fgsm /= trials
    fss_svm_fgsm  /= trials
    tprs_svm_fgsm /= trials
    tnrs_svm_fgsm /= trials
    mccs_svm_fgsm /= trials
    accs_ee_fgsm  /= trials
    fss_ee_fgsm  /= trials
    tprs_ee_fgsm /= trials
    tnrs_ee_fgsm /= trials
    mccs_ee_fgsm /= trials
    accs_lo_fgsm /= trials
    fss_lo_fgsm /= trials
    tprs_lo_fgsm /= trials
    tnrs_lo_fgsm /= trials
    mccs_lo_fgsm /= trials

    accs_if_pgd /= trials 
    fss_if_pgd /= trials
    tprs_if_pgd /= trials
    tnrs_if_pgd /= trials
    mccs_if_pgd /= trials
    accs_svm_pgd /= trials
    fss_svm_pgd  /= trials
    tprs_svm_pgd /= trials
    tnrs_svm_pgd /= trials
    mccs_svm_pgd /= trials
    accs_ee_pgd  /= trials
    fss_ee_pgd  /= trials
    tprs_ee_pgd /= trials
    tnrs_ee_pgd /= trials
    mccs_ee_pgd /= trials
    accs_lo_pgd /= trials
    fss_lo_pgd /= trials
    tprs_lo_pgd /= trials
    tnrs_lo_pgd /= trials
    mccs_lo_pgd /= trials

    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')

    np.savez(OUTPUT_FILE,
             accs_if_baseline = accs_if_baseline, 
             fss_if_baseline = fss_if_baseline,
             tprs_if_baseline = tprs_if_baseline,
             tnrs_if_baseline = tnrs_if_baseline,
             mccs_if_baseline = mccs_if_baseline,
             accs_svm_baseline = accs_svm_baseline,
             fss_svm_baseline  = fss_svm_baseline,
             tprs_svm_baseline = tprs_svm_baseline,
             tnrs_svm_baseline = tnrs_svm_baseline,
             mccs_svm_baseline = mccs_svm_baseline,
             accs_ee_baseline = accs_ee_baseline,
             fss_ee_baseline = fss_ee_baseline,
             tprs_ee_baseline = tprs_ee_baseline,
             tnrs_ee_baseline = tnrs_ee_baseline,
             mccs_ee_baseline = mccs_ee_baseline,
             accs_lo_baseline = accs_lo_baseline,
             fss_lo_baseline = fss_lo_baseline,
             tprs_lo_baseline = tprs_lo_baseline,
             tnrs_lo_baseline = tnrs_lo_baseline,
             mccs_lo_baseline = mccs_lo_baseline,
             accs_if_deepfool = accs_if_deepfool, 
             fss_if_deepfool = fss_if_deepfool,
             tprs_if_deepfool = tprs_if_deepfool,
             tnrs_if_deepfool = tnrs_if_deepfool,
             mccs_if_deepfool = mccs_if_deepfool,
             accs_svm_deepfool = accs_svm_deepfool,
             fss_svm_deepfool  = fss_svm_deepfool,
             tprs_svm_deepfool = tprs_svm_deepfool,
             tnrs_svm_deepfool = tnrs_svm_deepfool,
             mccs_svm_deepfool = mccs_svm_deepfool,
             accs_ee_deepfool = accs_ee_deepfool,
             fss_ee_deepfool = fss_ee_deepfool,
             tprs_ee_deepfool = tprs_ee_deepfool,
             tnrs_ee_deepfool = tnrs_ee_deepfool,
             mccs_ee_deepfool = mccs_ee_deepfool,
             accs_lo_deepfool = accs_lo_deepfool,
             fss_lo_deepfool = fss_lo_deepfool,
             tprs_lo_deepfool = tprs_lo_deepfool,
             tnrs_lo_deepfool = tnrs_lo_deepfool,
             mccs_lo_deepfool = mccs_lo_deepfool,
             accs_if_fgsm = accs_if_fgsm, 
             fss_if_fgsm = fss_if_fgsm,
             tprs_if_fgsm = tprs_if_fgsm,
             tnrs_if_fgsm = tnrs_if_fgsm,
             mccs_if_fgsm = mccs_if_fgsm,
             accs_svm_fgsm = accs_svm_fgsm,
             fss_svm_fgsm  = fss_svm_fgsm,
             tprs_svm_fgsm = tprs_svm_fgsm,
             tnrs_svm_fgsm = tnrs_svm_fgsm,
             mccs_svm_fgsm = mccs_svm_fgsm,
             accs_ee_fgsm = accs_ee_fgsm,
             fss_ee_fgsm = fss_ee_fgsm,
             tprs_ee_fgsm = tprs_ee_fgsm,
             tnrs_ee_fgsm = tnrs_ee_fgsm,
             mccs_ee_fgsm = mccs_ee_fgsm,
             accs_lo_fgsm = accs_lo_fgsm,
             fss_lo_fgsm = fss_lo_fgsm,
             tprs_lo_fgsm = tprs_lo_fgsm,
             tnrs_lo_fgsm = tnrs_lo_fgsm,
             mccs_lo_fgsm = mccs_lo_fgsm,
             accs_if_pgd = accs_if_pgd, 
             fss_if_pgd = fss_if_pgd,
             tprs_if_pgd = tprs_if_pgd,
             tnrs_if_pgd = tnrs_if_pgd,
             mccs_if_pgd = mccs_if_pgd,
             accs_svm_pgd = accs_svm_pgd,
             fss_svm_pgd  = fss_svm_pgd,
             tprs_svm_pgd = tprs_svm_pgd,
             tnrs_svm_pgd = tnrs_svm_pgd,
             mccs_svm_pgd = mccs_svm_pgd,
             accs_ee_pgd = accs_ee_pgd,
             fss_ee_pgd = fss_ee_pgd,
             tprs_ee_pgd = tprs_ee_pgd,
             tnrs_ee_pgd = tnrs_ee_pgd,
             mccs_ee_pgd = mccs_ee_pgd,
             accs_lo_pgd = accs_lo_pgd,
             fss_lo_pgd = fss_lo_pgd,
             tprs_lo_pgd = tprs_lo_pgd,
             tnrs_lo_pgd = tnrs_lo_pgd,
             mccs_lo_pgd = mccs_lo_pgd, 
             accs_if_dt = accs_if_dt, 
             fss_if_dt = fss_if_dt,
             tprs_if_dt = tprs_if_dt,
             tnrs_if_dt = tnrs_if_dt,
             mccs_if_dt = mccs_if_dt,
             accs_svm_dt = accs_svm_dt,
             fss_svm_dt  = fss_svm_dt,
             tprs_svm_dt = tprs_svm_dt,
             tnrs_svm_dt = tnrs_svm_dt,
             mccs_svm_dt = mccs_svm_dt,
             accs_ee_dt = accs_ee_dt,
             fss_ee_dt = fss_ee_dt,
             tprs_ee_dt = tprs_ee_dt,
             tnrs_ee_dt = tnrs_ee_dt,
             mccs_ee_dt = mccs_ee_dt,
             accs_lo_dt = accs_lo_dt,
             fss_lo_dt = fss_lo_dt,
             tprs_lo_dt = tprs_lo_dt,
             tnrs_lo_dt = tnrs_lo_dt,
             mccs_lo_dt = mccs_lo_dt
    )

    return None 


def run_experiment_causative(dataset:str='unswnb15', 
                               trials:int=10, 
                               type:str='attacks_causative', 
                               verbose:bool=False): 
    """TBD
    """
    return None 
