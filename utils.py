#!/usr/bin/env python 

import numpy as np 


def get_performance(y_true:np.ndarray, 
                    y_hat:np.ndarray, 
                    verbatim:bool,
                    pos:float=-1.0, 
                    neg:float=1.0): 
    """Calculate the performance of a detector / classifier
    """    
    tp, tn, fp, fn = 0., 0., 0., 0.
    
    for yt, yh in zip(y_true, y_hat): 
        if yt == neg and yh == neg: 
            tn += 1.
        elif yt == pos and yh == pos: 
            tp += 1.
        elif yt == neg and yh == pos: 
            fp += 1.
        elif yt == pos and yh == neg: 
            fn += 1.
    
    acc = (tp+tn)/(fp+fn+tp+tn)
    fs = tp/(tp+0.5+(fp+fn))
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    mcc = tp*tn/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    if verbatim: 
        print(''.join([' Accuracy:  ', str(acc*100)]))
        print(''.join([' F-score:   ', str(fs*100)]))
        print(''.join([' TPR:       ', str(tpr*100)]))
        print(''.join([' TNR:       ', str(tnr*100)]))
        print(''.join([' MCC:       ', str(mcc*100)]))
    return acc, fs, tpr, tnr, mcc
