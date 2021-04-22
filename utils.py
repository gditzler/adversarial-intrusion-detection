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
                    pos:float=-1.0, 
                    neg:float=1.0): 
    """Calculate the performance of a detector / classifier

    :param y_true: numpy array with the true class labels [required]
    :param y_hat: numpy array with the predicted class labels [required]
    :param pos: label of the positive class
    :param neg: label of the negative class
    :return tuple[acc, fs, tpr, tnr, mcc] 
    """    
    tp, tn, fp, fn = 0., 0., 0., 0.
    
    # get the tp, tn, fp, and fn
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
    if tp+0.5+(fp+fn) == 0: 
        fs = 0.
    else: 
        fs = tp/(tp+0.5+(fp+fn))

    if tp+fn == 0: 
        tpr = 0. 
    else: 
        tpr = tp/(tp+fn)

    if tn+fp == 0: 
        tnr = 0.
    else: 
        tnr = tn/(tn+fp)
    
    if tp+fp==0 or tp+fn==0 or tn+fp==0 or tn+fn==0: 
        mcc = 0.
    else: 
        mcc = tp*tn/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    return acc, fs, tpr, tnr, mcc
