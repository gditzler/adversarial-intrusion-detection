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

    Xtr, ytr, Xte, yte = load_dataset(name='nslkdd')

    X = Xtr #[:100]
    y = ytr #[:100]

    Xadv, yadv = generate_causative_adversarial_data(X_tr=Xtr, 
                                                     y_tr=ytr, 
                                                     X=X, 
                                                     y=y,
                                                     max_iter=5,
                                                     pp_poison=0.95,
                                                     atype='cleanlabel_pattern')
    np.savez_compressed('data/causative/full_data_nslkdd_cleanlabel_pattern.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xadv, yaml=yadv)

    Xadv, yadv = generate_causative_adversarial_data(X_tr=Xtr, 
                                                     y_tr=ytr, 
                                                     X=X, 
                                                     y=y,
                                                     max_iter=5,
                                                     pp_poison=0.8,
                                                     atype='cleanlabel_single')
    np.savez_compressed('data/causative/full_data_unswnb15_cleanlabel_single.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xadv, yaml=yadv)
    
    
    # Xtr, ytr, Xte, yte = load_dataset(name='unswnb15')

    # X = Xtr #[:100]
    # y = ytr #[:100]

    # Xadv, yadv = generate_causative_adversarial_data(X_tr=Xtr, 
    #                                                  y_tr=ytr, 
    #                                                  X=X, 
    #                                                  y=y,
    #                                                  max_iter=5,
    #                                                  pp_poison=0.2,
    #                                                  atype='cleanlabel_pattern')
    # np.savez_compressed('data/causative/full_data_unswnb15_cleanlabel_pattern.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xadv, yaml=yadv)

    # Xadv, yadv = generate_causative_adversarial_data(X_tr=Xtr, 
    #                                                  y_tr=ytr, 
    #                                                  X=X, 
    #                                                  y=y,
    #                                                  max_iter=5,
    #                                                  pp_poison=0.2,
    #                                                  atype='cleanlabel_single')
    # np.savez_compressed('data/causative/full_data_unswnb15_cleanlabel_single.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, Xaml=Xadv, yaml=yadv)

