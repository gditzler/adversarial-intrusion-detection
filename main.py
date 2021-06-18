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


from experiments import run_experiment_exploratory, run_experiment_causative

TRIALS = 15 
VERBOSE = True 

if __name__ == '__main__': 
    # run the experiments on the nslkdd dataset
    DATASET = 'nslkdd' 
    run_experiment_exploratory(dataset=DATASET, trials=TRIALS, type='attacks_all', verbose=VERBOSE)
    run_experiment_exploratory(dataset=DATASET, trials=TRIALS, type='attacks_only', verbose=VERBOSE) 
    run_experiment_causative(dataset=DATASET, trials=TRIALS, ppoison=0.05, verbose=VERBOSE)
    run_experiment_causative(dataset=DATASET, trials=TRIALS, ppoison=0.1, verbose=VERBOSE)
    run_experiment_causative(dataset=DATASET, trials=TRIALS, ppoison=0.15, verbose=VERBOSE)
