from jax.scipy.linalg import cholesky, solve_triangular
from .utils import add_to_diagonal
from copy import deepcopy
from typing import Callable
import jax.numpy as jnp


class SGPVFE:
    kernel : Callable
    obs_noise : Array
        
    def trace_term(self, X):
        Kffdiag = self.kernel.diag(X)
        Qffdiag = jnp.power(self.W, 2).sum(axis=1)
        trace_term = (Kffdiag - Qffdiag).sum() / self.obs_noise
        trace_term = jnp.clip(trace_term, a_min=0.0) 
        return -trace_term / 2.0
        
def vfe_precompute(X, X_u, obs_noise, kernel, jitter: float=1e-5):
    
    # Kernel
    Kuu = kernel.gram(X_u)
    Kuu = add_to_diagonal(Kuu, jitter)
    Luu = cholesky(Kuu, lower=True)
    
    Kuf = kernel.cross_covariance(X_u, X)
    
    
    # calculate cholesky
    Luu = cholesky(Kuu, lower=True)
    
    # compute W
    W = solve_triangular(Luu, Kuf, lower=True).T
    
    # compute D
    D = jnp.ones(Kuf.shape[1]) * obs_noise
    
    return Luu, W, D


def get_cond_params(
    kernel, params: dict, x: Array, y: Array, jitter: float = 1e-5
) -> dict:

    params = deepcopy(params)
    x_u = params.pop("x_u")
    obs_noise = params.pop("obs_noise")
    kernel = kernel(**params)
    n_samples = x.shape[0]

    # calculate the cholesky factorization
    Luu, W, D = vfe_precompute(x, x_u, obs_noise, kernel, jitter=jitter)

    W_Dinv = W.T / D
    K = W_Dinv @ W
    K = add_to_diagonal(K, 1.0)
    L = cholesky(K, lower=True)

    # mean function
    y_residual = y  # mean function
    y_2D = y_residual.reshape(-1, n_samples).T
    W_Dinv_y = W_Dinv @ y_2D

    return {
        "X": x,
        "y": y,
        "Luu": Luu,
        "L": L,
        "W_Dinv_y": W_Dinv_y,
        "x_u": x_u,
        "kernel_params": params,
        "obs_noise": obs_noise,
        "kernel": kernel,
    }