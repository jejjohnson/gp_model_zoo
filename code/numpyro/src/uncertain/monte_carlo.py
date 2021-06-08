from typing import Tuple

from .moment import MomentTransform
import jax
from chex import Array, dataclass
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions


@dataclass
class MCMomentTransform(MomentTransform):
    n_features: int
    n_samples: int
    seed: int

    def __post_init__(
        self,
    ):
        self.rng_key = jr.PRNGKey(self.seed)
        self.z = dist.Normal(
            loc=jnp.zeros((self.n_features,)), scale=jnp.ones(self.n_features)
        )
        wm, wc = get_mc_weights(self.n_samples)
        self.wm, self.wc = wm, wc

    def predict_f(self, f, x, x_cov, rng_key):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)
        # print(x_mc_samples.shape, y_mu_mc.shape)

        # mean of mc samples
        # (P,) = (P,M)

        y_mu = jnp.mean(y_mu_mc, axis=1)
        # print(y_mu.shape, y_mu_mc.shape)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_mc - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_var = jnp.diag(self.wc * dfydx @ dfydx.T)

        y_var = jnp.atleast_1d(y_var)

        return y_mu, y_var

    def mean(self, f, x, x_cov):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), self.rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)
        # print(y_mu_mc.shape, x_mc_samples.shape)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)
        # print(y_mu.shape, y_mu_mc.shape)

        return y_mu

    def covariance(self, f, x, x_cov):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), self.rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_mc - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_cov = self.wc * dfydx @ dfydx.T

        return y_cov

    def variance(self, f, x, x_cov):

        y_cov = self.covariance(f, x, x_cov)

        y_var = jnp.diag(y_cov)

        y_var = jnp.atleast_1d(y_var)

        return y_var


def get_mc_weights(n_samples: int = 100) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""
    mean = 1.0 / n_samples
    cov = 1.0 / (n_samples - 1)
    return mean, cov


def get_mc_sigma_points(rng_key, n_features: int, n_samples: int = 100) -> Array:
    """Generate MCMC samples from a normal distribution"""

    sigma_dist = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))

    return sigma_dist.sample((n_samples,), rng_key)
