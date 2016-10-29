'''
Script to sample from learnt basis and LAHMC
Mayur Mudigonda
'''

from LAHMC import LAHMC
import numpy as np
import theano.tensor as T
import theano

def E(a):
    E_val =  np.dot(basis,a) + np.sum(np.abs(a),axis=0)
    return E_val

def dEdX(X):

    return E_val
