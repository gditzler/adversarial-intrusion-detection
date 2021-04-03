#!/usr/bin/env python 

import numpy as np 


def accs(y_true, y_hat, verb): 
    """
    """
    
    tp, tn, fp, fn = 0., 0., 0., 0.
    
    for yt, yh in zip(y_true, y_hat): 
        if yt == 0 and yh == 0: 
            tn += 1
        elif yt == 1 and yh == 1: 
            tp += 1
        elif yt == 0 and yh == 1: 
            fp += 1
        elif yt == 1 and yh == 0: 
            fn += 1 
    
    acc = (tp+tn)/(fp+fn+tp+tn)
    fs = tp/(tp+0.5+(fp+fn))
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    mcc = tp*tn/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    if verb: 
        print(''.join([' Accuracy:  ', str(acc*100)]))
        print(''.join([' F-score:   ', str(fs*100)]))
        print(''.join([' TPR:       ', str(tpr*100)]))
        print(''.join([' TNR:       ', str(tnr*100)]))
        print(''.join([' MCC:       ', str(mcc*100)]))
    return acc, fs, tpr, tnr, mcc
