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
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

#%% Get some sample data

n_samples = 500
n_features = 50
random_state = 123

X, y = make_regression(n_samples=n_samples,
                       n_features=n_features,
                       random_state=random_state)



kernel = C() * RBF() + WhiteKernel()
gp_model = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

gp_model.fit(X, y)

#%%
y_pred, sigma = gp_model.predict(X, return_std=True)


signal_variance = gp_model.kernel_.get_params()['k1__k1__constant_value']
length_scale = gp_model.kernel_.get_params()['k1__k2__length_scale']
noise_likelihood = gp_model.kernel_.get_params()['k2__noise_level']

#%%

fig, ax = plt.subplots()
ax.plot(y_pred, 'r')
ax.plot(sigma)
ax.legend()
plt.show()



#%%
# =============================================================================
# Standard Parameter Retrieval
# =============================================================================

x, y = X, y

n_samples, d_dimensions = x.shape


signal_variance = gp_model.kernel_.get_params()['k1__k1__constant_value']
length_scale = 1/gp_model.kernel_.get_params()['k1__k2__length_scale']**2
length_scale = length_scale * np.ones(shape=X.shape[1]).squeeze()
noise_likelihood = gp_model.kernel_.get_params()['k2__noise_level']

# Initialize Sigma X
SigmaX = 0.1 * np.ones(shape=X.shape[1])
SigmaX = np.diag(SigmaX)


#%% Initialize the covariance Matrix
K = gp_model.kernel_(X)


# Calculate the Inverse
Kinv = np.linalg.inv(K)

#%% 
# =============================================================================
# Error Propagation
# =============================================================================

mu = np.zeros(y_pred.shape)
S2 = np.zeros(sigma.shape)

t0 = time.time()

for iteration, ix in enumerate(X):
    
    xtest = ix
    if np.ndim(xtest) == 1:
        xtest = np.atleast_2d(ix)
    

    # Covariance (X, X*)
    K_traintest = gp_model.kernel_(X, xtest)
    
    # Covariance (X*, X*)
    Ktest = signal_variance + noise_likelihood
    
    # Predicted Mean
    Kinvt = np.dot(Kinv, y[:, np.newaxis])
    mu[iteration] = np.dot(K_traintest.T, Kinvt)
    
    # Predicted Variance
    S2[iteration] = Ktest - np.sum(K_traintest * (Kinv.dot(K_traintest)))
    
    
    # More Error Propgation
    deriv = np.zeros(shape=(1, d_dimensions))
    S2deriv = np.zeros(shape=(d_dimensions, d_dimensions))
    
    for idim in range(d_dimensions):
        tmp1 = X[:,idim]
        tmp2 = xtest[:, idim]
        tmp3 = tmp1 - tmp2
        c = (K_traintest.flatten() *  tmp3.flatten())[:, np.newaxis]
        deriv[:, idim] = length_scale[idim] * c.T.dot(Kinvt)
        ainvKc = K_traintest * np.dot(Kinv, c)
        
        for jdim in range(d_dimensions):
            exp_t = - length_scale[idim] * length_scale[jdim]
            tmp1 = X[:, jdim] 
            tmp2 = xtest[:, jdim]
            tmp3 = (tmp1 - tmp2)[:, np.newaxis]
            S2deriv[idim, jdim] = \
                 exp_t * np.sum(ainvKc * tmp3) 
        
        S2deriv[idim, idim] = S2deriv[idim, idim] + length_scale[idim] * signal_variance
    
    S2[iteration] += deriv.dot(SigmaX).dot(deriv.T) + 0.5 * np.sum(np.sum(SigmaX * S2deriv))
    
    

t1 = time.time() - t0
#%%
    
fig, ax = plt.subplots()
ax.plot(mu, 'r')
ax.plot(S2)
ax.legend()
plt.show()

















