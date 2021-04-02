#!/usr/bin/env python 


import os

from experiments import run_experiment


if __name__ == '__main__': 
    run_experiment(dataset='unswnb15', ids='isofor', trials=10, verbose=True) 
    print('Done')