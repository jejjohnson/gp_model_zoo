# -*- coding: utf-8 -*-
"""
This is a script that will attempt to do some error propagation calculations.
There is a library found at this link: 
    https://github.com/Dynamic-Systems-and-GP/GPdyn

I will try to implement a few of the algorithms to see if I can get something
from it.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel as C,
                                              RBF, WhiteKernel)
import scipy.io as sio
file_location = '/home/emmanuel/code/matlab_stuff/gp_error/GPdyn/gpdyn-demos/'

#%% Get some sample data
example_file = sio.loadmat(file_location + 'example_data.mat')
example_trained_file = sio.loadmat(file_location + 'example_trained.mat')

inputs = example_trained_file['input']
targets = example_trained_file['target']
test = np.hstack([np.zeros(example_file['uvalid'].shape), example_file['uvalid']])
tmp = example_file['uvalid']


#%%



# =============================================================================
# Standard Parameter Retrieval
# =============================================================================

x, y = inputs, targets

n_samples, d_dimensions = x.shape



#%% Get Hyper Parameters

# Get the best LOG parameters in from the training
theta = np.hstack([-1.2918, -0.1168, 2.4252, -5.8460])

exptheta = np.exp(theta)        # exponentiate hyperparameters

# extract noise likelihood
noise_likelihood = np.exp(theta[-1])
length_scale = np.exp(theta[:d_dimensions])
signal_variance = np.exp(theta[d_dimensions])
# Initialize Sigma X
SigmaX = np.zeros(shape=(d_dimensions, d_dimensions))


#%%

def ard_kernel(X, Y, length_scale, signal_variance):
    n, d = X.shape
    nn, d = Y.shape
    K = np.zeros(shape=(n, nn))
    
    for id in range(d):
        tm1 = np.tile(X[:, id][:, np.newaxis], (1, nn))
        tm2 = np.tile(Y[:, id][:, np.newaxis].T, (n, 1))
        K = K + length_scale[id]*(tm1 - tm2)**2 

    return signal_variance * np.exp(-0.5 * K)


#%% Initialize the covariance Matrix
K = ard_kernel(inputs, inputs, length_scale, signal_variance)

# Add the noise likelihood
K[np.diag_indices_from(K)] += noise_likelihood

# Calculate the Inverse
Kinv = np.linalg.inv(K)

#%% 
# =============================================================================
# Error Propagation
# =============================================================================


xtest = np.atleast_2d(test[0, :])


# Covariance (X, X*)
K_traintest = ard_kernel(inputs, xtest, length_scale, signal_variance)

# Covariance (X*, X*)
Ktest = signal_variance + noise_likelihood

# Predicted Mean
Kinvt = np.dot(Kinv, y)
mu = np.dot(K_traintest.T, Kinvt)

# Predicted Variance
sig2 = Ktest - np.sum(K_traintest * (Kinv.dot(K_traintest)))
S2 = sig2


#%% More Error Propgation

deriv = np.zeros(shape=(1, d_dimensions))
S2deriv = np.zeros(shape=(d_dimensions, d_dimensions))

for idim in range(d_dimensions):
    tmp1 = inputs[:,idim]
    tmp2 = xtest[:, idim]
    tmp3 = tmp1 - tmp2
    c = (K_traintest.flatten() *  tmp3.flatten())[:, np.newaxis]
    deriv[:, idim] = length_scale[idim] * c.T.dot(Kinvt)
    ainvKc = K_traintest * np.dot(Kinv, c)
    
    for jdim in range(d_dimensions):
        exp_t = - length_scale[idim] * length_scale[jdim]
        tmp1 = inputs[:, jdim] 
        tmp2 = xtest[:, jdim]
        tmp3 = (tmp1 - tmp2)[:, np.newaxis]
        S2deriv[idim, jdim] = \
             exp_t * np.sum(ainvKc * tmp3) 
    
    S2deriv[idim, idim] = S2deriv[idim, idim] + length_scale[idim] * signal_variance

S2 = sig2 + deriv.dot(SigmaX).dot(deriv.T) + 0.5 * np.sum(np.sum(SigmaX * S2deriv))





















