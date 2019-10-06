# -*- coding: utf-8 -*-
"""
This is a script that will attempt to do some error propagation calculations.
There is a library found at this link: 
    https://github.com/Dynamic-Systems-and-GP/GPdyn

I will try to implement a few of the algorithms to see if I can get something
from it.
"""
from data import example_1d
import numpy as np
from sklearn.datasets import make_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel as C,
                                              RBF, WhiteKernel)
import scipy.io as sio
import matplotlib.pyplot as plt
file_location = '/home/emmanuel/code/matlab_stuff/gp_error/GPdyn/gpdyn-demos/'

#%% Get some sample data

X, y, error_params = example_1d()

Xtrain, Xtest = X['train'], X['test']
ytrain, ytest = y['train'], y['test']
Xplot, yplot = X['plot'], y['plot']

kernel = C() * RBF() + WhiteKernel()
gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

gp_model.fit(Xtrain, ytrain)
y_pred, sigma = gp_model.predict(Xplot, return_std=True)


signal_variance = gp_model.kernel_.get_params()['k1__k1__constant_value']
length_scale = gp_model.kernel_.get_params()['k1__k2__length_scale']
noise_likelihood = gp_model.kernel_.get_params()['k2__noise_level']

#%%

fig, ax = plt.subplots()

ax.plot(Xplot, yplot, 'r:', label='f(x)')
ax.plot(Xtrain, ytrain, 'r.', markersize=10, label=u'Observations')
ax.plot(Xplot, y_pred, 'b-', label=u'Predictions')
plt.fill_between(Xplot[:,0],
                 y_pred.flatten() - 1.9600 * sigma.flatten() ,
                 y_pred.flatten()  + 1.9600 * sigma.flatten() ,
                 alpha=0.25, color='darkorange')
ax.legend()
plt.show()



#%%
# =============================================================================
# Standard Parameter Retrieval
# =============================================================================

inputs, targets = Xtrain, ytrain

n_samples, d_dimensions = inputs.shape


signal_variance = gp_model.kernel_.get_params()['k1__k1__constant_value']
length_scale = 1/gp_model.kernel_.get_params()['k1__k2__length_scale']**2
noise_likelihood = gp_model.kernel_.get_params()['k2__noise_level']

# Initialize Sigma X
SigmaX = 0.3 #error_params['x']
SigmaX = np.diag(np.array([SigmaX]))

length_scale = np.array([length_scale])

#%% Initialize the covariance Matrix
K = gp_model.kernel_(inputs)

beta = gp_model.alpha_
# Calculate the Inverse
Kinv = np.linalg.inv(K)

Ktest = signal_variance + noise_likelihood


#%% 
# =============================================================================
# Error Propagation
# =============================================================================

mu = np.zeros(y_pred.shape)
S2 = np.zeros(sigma.shape)

for iteration, ix in enumerate(Xplot):


    #%%
    SigX = SigmaX
    muX = np.atleast_2d(ix)
    invL = np.diag(length_scale)
    invS = np.diag(np.linalg.inv(SigX))
    invC = invL + invS
    invSmuX = np.dot(invS, muX.T)
    t1 = np.dot(muX, invSmuX)
    c = np.linalg.inv(invC).dot(invL.dot(inputs.T) + np.tile(invSmuX, (1, n_samples)))
    #c = np.linalg.inv(invC).dot()
    t2 = np.sum(inputs * (inputs.dot(invL)), axis=1)[:, np.newaxis]
    t3 = np.sum(c * np.dot(invC, c), axis=0).T[:, np.newaxis]
    I = 1 / np.sqrt(np.linalg.det(np.dot(invL, SigX) + np.eye(d_dimensions)))
    I *= np.exp(-.5 * (t1 + t2 + t3))
    mu[iteration] = Ktest * np.dot(I.T, beta)
    
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
    S2[iteration] = Ktest - Ktest**2 * np.sum(np.sum((Kinv - beta.dot(beta.T)) * I2 )) \
        - mu[iteration]**2 + noise_likelihood

#%%
    
fig, ax = plt.subplots()

ax.plot(Xplot, yplot, 'r:', label='f(x)')
ax.plot(Xtrain, ytrain, 'r.', markersize=10, label=u'Observations')
ax.plot(Xplot, mu, 'b-', label=u'Predictions')    
plt.fill_between(Xplot[:,0],
                 mu.flatten() - 1.9600 * np.sqrt(S2).flatten() ,
                 mu.flatten()  + 1.9600 * np.sqrt(S2).flatten() ,
                 alpha=0.25, color='darkorange')    
ax.legend()
plt.show()
















