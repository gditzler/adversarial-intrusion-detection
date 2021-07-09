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
from data import generate_causative_adversarial_data, load_dataset

if __name__ == '__main__': 

    Xtr, ytr, Xte, yte = load_dataset(name='awid')
    q = int(.15*len(ytr))
    n = np.random.randint(0, len(ytr), 500)
    m = np.random.randint(0, len(ytr), q) 
    X = Xtr[m] #[q:]
    y = ytr[m] #[q:]

    Xadv, yadv = generate_causative_adversarial_data(X_tr=Xtr[n], 
                                                     y_tr=ytr[n], 
                                                     X=X, 
                                                     y=y,
                                                     atype='svm')
    np.savez_compressed('data/causative/full_data_awid_svc.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xadv, yaml=yadv)

