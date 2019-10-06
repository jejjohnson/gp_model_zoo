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
SigmaX = np.array([[0.0032, 0.0], [0.0, 0.001]])

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
beta = np.dot(Kinv, y)
mu = np.dot(K_traintest.T, beta)

# Predicted Variance
sig2 = Ktest - np.sum(K_traintest * (Kinv.dot(K_traintest)))
S2 = sig2

#%%


SigX = SigmaX
muX = xtest
inputs = x
invL = np.diag(length_scale)
invS = np.diag(1/np.diag(SigX))
invC = invL + invS
invSmuX = np.dot(invS, muX.T)
t1 = np.dot(muX, invSmuX)
c = np.linalg.solve(invC, invL.dot(inputs.T) + np.tile(invSmuX, (1, n_samples)))
#c = np.linalg.inv(invC).dot()
t2 = np.sum(inputs * (inputs.dot(invL)), axis=1)[:, np.newaxis]
t3 = np.sum(c * np.dot(invC, c), axis=0).T[:, np.newaxis]
I = (1 / np.sqrt(np.linalg.det(np.dot(invL, SigX) + np.eye(d_dimensions))))
I *= np.exp(-.5 * (t1 + t2 + t3))
m = Ktest * np.dot(I.T, beta)

invD = 2 * invL + invS
[kk1, kk2] = np.meshgrid(range(n_samples), range(n_samples))
T1 = np.tile(inputs, (n_samples, 1)) + inputs[kk1.flatten(), :]
invLT1 = np.dot(T1, invL)
d = np.linalg.solve(invD, invLT1.T + np.tile(invSmuX, (1, n_samples**2)))
T3 = np.reshape(np.sum(d * np.dot(invD, d), axis=0), (n_samples, n_samples ))
I2 = (1 / np.sqrt(np.linalg.det(np.dot(invL, SigX) + np.eye(d_dimensions))))
I2 *= np.exp(-0.5 * (t1 \
                     + np.tile(t2, (1, n_samples)) \
                     + np.tile(t2.T, (n_samples, 1)) \
                     - T3)) 
S2 = Ktest - Ktest**2 * np.sum(np.sum((Kinv - beta.dot(beta.T)) * I2 )) \
    - m**2 + noise_likelihood

#%%


















