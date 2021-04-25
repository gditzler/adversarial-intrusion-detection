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
import tensorflow as tf
from data import load_dataset, generate_adversarial_data

tf.compat.v1.disable_eager_execution()


if __name__ == '__main__': 
    Xtr, ytr, Xte, yte = load_dataset(name='unswnb15')

    
    X = Xte 

    # ctype=dt, atype=dt
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='dt', atype='dt')
    np.savez_compressed('data/attacks_all/full_data_unswnb15_dt_dt.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=deepfool
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='deepfool')
    np.savez_compressed('data/attacks_all/full_data_unswnb15_mlp_deepfool.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=deepfool
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='fgsm')
    np.savez_compressed('data/attacks_all/full_data_unswnb15_mlp_fgsm.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=pgd
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='pgd')
    np.savez_compressed('data/attacks_all/full_data_unswnb15_mlp_pgd.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # prepare the adversarial / normal datasets with the cyber attack data as being adversarial 
    X = Xte[yte==1]

    # ctype=dt, atype=dt
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='dt', atype='dt')
    np.savez_compressed('data/attacks_only/full_data_unswnb15_dt_dt.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=deepfool
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='deepfool')
    np.savez_compressed('data/attacks_only/full_data_unswnb15_mlp_deepfool.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=deepfool
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='fgsm')
    np.savez_compressed('data/attacks_only/full_data_unswnb15_mlp_fgsm.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)

    # ctype=mlp, atype=pgd
    Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='pgd')
    np.savez_compressed('data/attacks_only/full_data_unswnb15_mlp_pgd.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)



    # ctype=mlp, atype=cw takes a long time to run
    #Xaml = generate_adversarial_data(X_tr=Xtr, y_tr=ytr, X=X, ctype='mlp', atype='cw')
    #np.savez_compressed('data/full_data_unswnb15_mlp_cw.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xaml)