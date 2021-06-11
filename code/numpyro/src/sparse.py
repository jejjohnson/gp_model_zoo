from jax.scipy.linalg import cholesky, solve_triangular
from .utils import add_to_diagonal, init_inducing_kmeans
from .base import Predictive
from .means import zero_mean
from .kernels import RBF
from copy import deepcopy
from typing import Callable, Tuple
import jax.numpy as jnp
from chex import Array, dataclass
import numpyro
import numpyro.distributions as dist
from flax import struct


@struct.dataclass
class SGPFITC:
    X: Array
    y: Array
    X_u: Array
    mean: Callable
    kernel: Callable
    obs_noise: Array
    jitter: float

    def to_numpyro(self, y=None):

        f_loc = self.mean(self.X)

        _, W, D = fitc_precompute(
            self.X, self.X_u, self.obs_noise, self.kernel, jitter=self.jitter
        )
        # Sample y according SGP
        if y is not None:

            return numpyro.sample(
                "y",
                dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.ndim - 1),
                obs=self.y,
            )
        else:

            return numpyro.sample(
                "y", dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D)
            )


@struct.dataclass
class SGPVFE:
    X: Array
    y: Array
    X_u: Array
    mean: Callable
    kernel: Callable
    obs_noise: Array
    jitter: float

    def trace_term(self, X, W, obs_noise):
        Kffdiag = self.kernel.diag(X)
        Qffdiag = jnp.power(W, 2).sum(axis=1)
        trace_term = (Kffdiag - Qffdiag).sum() / obs_noise
        trace_term = jnp.clip(trace_term, a_min=0.0)
        return -trace_term / 2.0

    def to_numpyro(self, y=None):

        f_loc = self.mean(self.X)

        _, W, D = vfe_precompute(
            self.X, self.X_u, self.obs_noise, self.kernel, jitter=self.jitter
        )

        numpyro.factor("trace_term", self.trace_term(self.X, W, self.obs_noise))
        # Sample y according SGP
        if y is not None:

            return numpyro.sample(
                "y",
                dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D)
                .expand_by(self.y.shape[:-1])
                .to_event(self.y.ndim - 1),
                obs=self.y,
            )
        else:

            return numpyro.sample(
                "y", dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D)
            )


@struct.dataclass
class SGPPredictive(Predictive):
    X: Array
    y: Array
    x_u: Array
    Luu: Array
    L: Array
    W_Dinv_y: Array
    obs_noise: dict
    kernel_params: dict
    kernel: Callable

    def _pred_factorize(self, xtest):

        Kux = self.kernel.cross_covariance(self.x_u, xtest)
        Ws = solve_triangular(self.Luu, Kux, lower=True)
        # pack
        pack = jnp.concatenate([self.W_Dinv_y, Ws], axis=1)
        Linv_pack = solve_triangular(self.L, pack, lower=True)
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, : self.W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, self.W_Dinv_y.shape[1] :]

        return Ws, Linv_W_Dinv_y, Linv_Ws

    def predict_mean(self, xtest):

        _, Linv_W_Dinv_y, Linv_Ws = self._pred_factorize(xtest)

        loc_shape = self.y.T.shape[:-1] + (xtest.shape[0],)
        loc = (Linv_W_Dinv_y.T @ Linv_Ws).reshape(loc_shape)

        return loc.T

    def predict_cov(self, xtest, noiseless=False):
        n_test_samples = xtest.shape[0]

        Ws, _, Linv_Ws = self._pred_factorize(xtest)

        Kxx = self.kernel.gram(xtest)

        if not noiseless:
            Kxx = add_to_diagonal(Kxx, self.obs_noise)

        Qss = Ws.T @ Ws

        Σ = Kxx - Qss + Linv_Ws.T @ Linv_Ws

        Σ_shape = self.y.T.shape[:-1] + (n_test_samples, n_test_samples)
        Σ = jnp.reshape(Σ, Σ_shape)

        return Σ.reshape([1, 2, 0])

    def _predict(self, xtest, full_covariance: bool = False, noiseless: bool = False):
        n_test_samples = xtest.shape[0]

        Ws, Linv_W_Dinv_y, Linv_Ws = self._pred_factorize(xtest)

        loc_shape = self.y.T.shape[:-1] + (xtest.shape[0],)
        μ = (Linv_W_Dinv_y.T @ Linv_Ws).reshape(loc_shape)

        if full_covariance:
            Kxx = self.kernel.gram(xtest)

            if not noiseless:
                Kxx = add_to_diagonal(Kxx, self.obs_noise)

            Qss = Ws.T @ Ws

            Σ = Kxx - Qss + Linv_Ws.T @ Linv_Ws

            Σ_shape = self.y.T.shape[:-1] + (n_test_samples, n_test_samples)
            Σ = jnp.reshape(Σ, Σ_shape)

            return μ, Σ.reshape([1, 2, 0])
        else:
            Kxx = self.kernel.diag(xtest)

            if not noiseless:
                Kxx += self.obs_noise

            Qss = jnp.power(Ws, 2).sum(axis=0)

            σ = Kxx - Qss + jnp.power(Linv_Ws, 2).sum(axis=0)

            cov_shape = self.y.T.shape[:-1] + (n_test_samples,)

            σ = jnp.reshape(σ, cov_shape)

            return μ.T, σ.T

    def predict_var(self, xtest, noiseless=False):

        n_test_samples = xtest.shape[0]

        Ws, _, Linv_Ws = self._pred_factorize(xtest)

        Kxx = self.kernel.diag(xtest)

        if not noiseless:
            Kxx += self.obs_noise

        Qss = jnp.power(Ws, 2).sum(axis=0)

        σ = Kxx - Qss + jnp.power(Linv_Ws, 2).sum(axis=0)

        cov_shape = self.y.T.shape[:-1] + (n_test_samples,)

        σ = jnp.reshape(σ, cov_shape)

        return σ.T


def init_default_sgp_model(
    Xtrain, n_features=1, inference="map", n_inducing=100, jitter=1e-5
):

    X_u_init = init_inducing_kmeans(Xtrain, n_inducing, seed=123)

    def numpyro_model(X, y):

        if inference == "map" or "vi_mf" or "vi_full":
            # Set priors on hyperparameters.
            η = numpyro.sample("variance", dist.HalfCauchy(scale=5.0))
            ℓ = numpyro.sample(
                "length_scale", dist.Gamma(2.0, 1.0), sample_shape=(n_features,)
            )
            σ = numpyro.sample("obs_noise", dist.HalfCauchy(scale=5.0))
        elif inference == "mll":

            # set params and constraints on hyperparams
            η = numpyro.param(
                "variance", init_value=1.0, constraints=dist.constraints.positive
            )
            ℓ = numpyro.param(
                "length_scale",
                init_value=jnp.ones(n_features),
                constraints=dist.constraints.positive,
            )
            σ = numpyro.param(
                "obs_noise", init_value=0.01, onstraints=dist.constraints.positive
            )
        else:
            raise ValueError(f"Unrecognized inference scheme: {inference}")

        x_u = numpyro.param("x_u", init_value=X_u_init)

        # Kernel Function
        rbf_kernel = RBF(variance=η, length_scale=ℓ)

        # GP Model
        gp_model = SGPVFE(
            X=X,
            X_u=x_u,
            y=y,
            mean=zero_mean,
            kernel=rbf_kernel,
            obs_noise=σ,
            jitter=jitter,
        )

        # Sample y according SGP
        return gp_model.to_numpyro(y=y)

    return numpyro_model


def init_sgp_predictive(kernel, params, x, y, jitter):

    return SGPPredictive(
        **get_cond_params(kernel=kernel, params=params, x=x, y=y, jitter=jitter)
    )


def vfe_precompute(X, X_u, obs_noise, kernel, jitter: float = 1e-5):

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


def fitc_precompute(X, X_u, obs_noise, kernel, jitter: float = 1e-5):

    # Kernel
    Kuu = kernel.gram(X_u)
    Kuu = add_to_diagonal(Kuu, jitter)
    Luu = cholesky(Kuu, lower=True)

    Kuf = kernel.cross_covariance(X_u, X)

    # calculate cholesky
    Luu = cholesky(Kuu, lower=True)

    # compute W
    W = solve_triangular(Luu, Kuf, lower=True).T

    Kffdiag = kernel.diag(X)
    Qffdiag = jnp.power(W, 2).sum(axis=1)
    D = Kffdiag - Qffdiag + obs_noise

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
