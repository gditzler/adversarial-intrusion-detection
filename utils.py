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
