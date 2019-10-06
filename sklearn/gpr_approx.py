#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 16:58:05 2018

@author: emmanuel
"""
import numpy as np
from gp_error.data import example_1d


    


def main():
    
    X, y, error_params = example_1d(func=1)
    
    Xtrain, Xtest = X['train'], X['test']
    ytrain, ytest = y['train'], y['test']
    Xplot, yplot = X['plot'], y['plot']

    
    pass

if __name__ == "__main__":
    main()
