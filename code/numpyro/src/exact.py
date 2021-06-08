from jax.scipy.linalg import cholesky, solve_triangular, cho_solve
from .utils import add_to_diagonal
from .base import Predictive
from copy import deepcopy
from typing import Callable, Tuple
import jax.numpy as jnp
from chex import Array, dataclass
import numpyro
import numpyro.distributions as dist
from flax import struct


@struct.dataclass
class GPRModel:
    X: Array
    y: Array
    mean: Callable
    kernel: Callable
    obs_noise: Array
    jitter: float

    def to_numpyro(self, y=None):

        f_loc = self.mean(self.X)

        Lff = precompute(self.X, self.obs_noise, self.kernel, jitter=self.jitter)
        # Sample y according SGP
        if y is not None:

            return numpyro.sample(
                "y",
                dist.MultivariateNormal(loc=f_loc, scale_tril=Lff)
                .expand_by(self.y.shape[:-1])  # for multioutput scenarios
                .to_event(self.y.ndim - 1),
                obs=self.y,
            )
        else:
            return numpyro.sample(
                "y",
                dist.MultivariateNormal(loc=jnp.zeros(self.X.shape[0]), scale_tril=Lff),
            )


def precompute(X, obs_noise, kernel, jitter):

    # Kernel
    Kff = kernel.gram(X)
    Kff = add_to_diagonal(Kff, obs_noise)
    Kff = add_to_diagonal(Kff, jitter)
    Lff = cholesky(Kff, lower=True)

    return Lff


@struct.dataclass
class GPRPredictive(Predictive):
    X: Array = struct.field(pytree_node=False)
    y: Array = struct.field(pytree_node=False)
    Lff: Array = struct.field(pytree_node=False)
    weights: Array = struct.field(pytree_node=False)
    obs_noise: Array = struct.field(pytree_node=False)
    kernel: Callable = struct.field(pytree_node=False)

    def predict_mean(self, xtest):

        K_x = self.kernel.cross_covariance(xtest, self.X)

        μ = jnp.dot(K_x, self.weights)

        return μ

    def predict_cov(self, xtest, noiseless=False):

        # Calculate the Mean
        K_x = self.kernel.cross_covariance(xtest, self.X)

        v = solve_triangular(self.Lff, K_x.T, lower=True)

        K_xx = self.kernel.gram(xtest)

        Σ = K_xx - v.T @ v

        if not noiseless:
            Σ = add_to_diagonal(Σ, self.obs_noise)

        return Σ

    def _predict(self, xtest, full_covariance: bool = False, noiseless: bool = True):

        # Calculate the Mean
        K_x = self.kernel.cross_covariance(xtest, self.X)
        μ = jnp.dot(K_x, self.weights)

        # calculate covariance
        v = solve_triangular(self.Lff, K_x.T, lower=True)

        if full_covariance:

            K_xx = self.kernel.gram(xtest)

            if not noiseless:
                K_xx = add_to_diagonal(K_xx, self.obs_noise)

            Σ = K_xx - v.T @ v

            return μ, Σ

        else:

            K_xx = self.kernel.diag(xtest)

            σ = K_xx - jnp.sum(jnp.square(v), axis=0)

            if not noiseless:
                σ += self.obs_noise

            return μ, σ

    def predict_var(self, xtest, noiseless=False):

        # Calculate the Mean
        K_x = self.kernel.cross_covariance(xtest, self.X)

        v = solve_triangular(self.Lff, K_x.T, lower=True)

        K_xx = self.kernel.diag(xtest)

        σ = K_xx - jnp.sum(jnp.square(v), axis=0)

        if not noiseless:
            σ += self.obs_noise

        return σ


def get_cond_params(
    kernel, params: dict, x: Array, y: Array, jitter: float = 1e-5
) -> dict:

    params = deepcopy(params)
    obs_noise = params.pop("obs_noise")
    kernel = kernel(**params)

    # calculate the cholesky factorization
    Lff = precompute(x, obs_noise, kernel, jitter=jitter)

    weights = cho_solve((Lff, True), y)

    return {
        "X": jnp.array(x),
        "y": jnp.array(y),
        "Lff": jnp.array(Lff),
        "obs_noise": jnp.array(obs_noise),
        "kernel": kernel,
        "weights": jnp.array(weights),
    }


def init_gp_predictive(
    kernel, params: dict, x: Array, y: Array, jitter: float = 1e-5
) -> dict:
    params = deepcopy(params)
    obs_noise = params.pop("obs_noise")
    kernel = kernel(**params)

    # calculate the cholesky factorization
    Lff = precompute(x, obs_noise, kernel, jitter=jitter)

    weights = cho_solve((Lff, True), y)

    def predict_mean(xtest):

        K_x = kernel.cross_covariance(xtest, x)

        μ = jnp.dot(K_x, weights)

        return μ

    def predict_cov(xtest, noiseless=False):

        # Calculate the Mean
        K_x = kernel.cross_covariance(xtest, x)

        v = solve_triangular(Lff, K_x.T, lower=True)

        K_xx = kernel.gram(xtest)

        Σ = K_xx - v.T @ v

        if not noiseless:
            Σ = add_to_diagonal(Σ, obs_noise)

        return Σ

    def _predict(xtest, full_covariance: bool = False, noiseless: bool = True):

        # Calculate the Mean
        K_x = kernel.cross_covariance(xtest, x)
        μ = jnp.dot(K_x, weights)

        # calculate covariance
        v = solve_triangular(Lff, K_x.T, lower=True)

        if full_covariance:

            K_xx = kernel.gram(xtest)

            if not noiseless:
                K_xx = add_to_diagonal(K_xx, obs_noise)

            Σ = K_xx - v.T @ v

            return μ, Σ

        else:

            K_xx = kernel.diag(xtest)

            σ = K_xx - jnp.sum(jnp.square(v), axis=0)

            if not noiseless:
                σ += obs_noise

            return μ, σ

    def predict_var(xtest, noiseless=False):

        # Calculate the Mean
        K_x = kernel.cross_covariance(xtest, x)

        v = solve_triangular(Lff, K_x.T, lower=True)

        K_xx = kernel.diag(xtest)

        σ = K_xx - jnp.sum(jnp.square(v), axis=0)

        if not noiseless:
            σ += obs_noise

        return σ

    def predict_f(xtest, full_covariance: bool = False):

        return _predict(xtest, full_covariance=full_covariance, noiseless=True)

    def predict_y(xtest, full_covariance: bool = False):

        return _predict(xtest, full_covariance=full_covariance, noiseless=False)

    return PredictTuple(
        predict_var=predict_var,
        predict_mean=predict_mean,
        predict_cov=predict_cov,
        predict_f=predict_f,
        predict_y=predict_y,
    )


from typing import NamedTuple


class PredictTuple(NamedTuple):
    predict_mean: Callable
    predict_var: Callable
    predict_cov: Callable
    predict_f: Callable
    predict_y: Callable
