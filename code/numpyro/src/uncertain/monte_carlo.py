from typing import Tuple

from .moment import MomentTransform
import jax
from chex import Array, dataclass
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions


def init_mc_transform(gp_pred, n_features: int, n_samples: int):

    z = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))

    wm, wc = get_mc_weights(n_samples)

    def predict_mean(key, x, x_cov):

        # sigma points
        sigma_pts = z.sample((n_samples,), key)

        # cholesky for input covariance (D,D)
        L = jnp.linalg.cholesky(x_cov)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=1)(x_mc_samples)
        # print(y_mu_mc.shape, x_mc_samples.shape)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)
        # print(y_mu.shape, y_mu_mc.shape)

        return y_mu

    def predict_f(key, x, x_cov, full_covariance=False):

        # sigma points
        sigma_pts = z.sample((n_samples,), key)
        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (N,P,M) = (D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=2)(x_mc_samples)

        # mean of mc samples
        # (N,P,) = (N,P,M)
        y_mu = jnp.mean(y_mu_mc, axis=2)

        if full_covariance:
            # ===================
            # Covariance
            # ===================
            # (N,P,M) - (N,P,1) -> (N,P,M)
            dfydx = y_mu_mc - y_mu[:, None]

            # (N,P,M) @ (N,P,M) -> (N,N,P)
            cov = wc * jnp.einsum("ijk,klm->imj", dfydx, dfydx.T)

            return y_mu, cov
        else:
            # (P,M) = (P,M) - (P, 1)
            var = jnp.var(y_mu_mc, axis=2)
            return y_mu, var

    def predict_cov(key, x, x_cov):

        # sigma points
        sigma_pts = z.sample((n_samples,), key)
        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (N,P,M) = (D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=2)(x_mc_samples)

        # mean of mc samples
        # (N,P,M) -> (N,P)
        y_mu = jnp.mean(y_mu_mc, axis=2)
        # ===================
        # Covariance
        # ===================
        # (N,P,M) - (N,P,1) -> (N,P,M)
        dfydx = y_mu_mc - y_mu[:, None]

        # (N,P,M) @ (N,P,M) -> (N,N,P)
        cov = wc * jnp.einsum("ijk,klm->imj", dfydx, dfydx.T)

        return cov

    def predict_var(key, x, x_cov):

        # sigma points
        sigma_pts = z.sample((n_samples,), key)
        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (N,P,M) = (D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=2)(x_mc_samples)

        # variance of mc samples
        # (N,P,M) -> (N,P)
        y_var = jnp.var(y_mu_mc, axis=2)

        return y_var

    return MomentTransform(
        predict_mean=predict_mean,
        predict_cov=predict_cov,
        predict_f=predict_f,
        predict_var=predict_var,
    )


def get_mc_weights(n_samples: int = 100) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""
    mean = 1.0 / n_samples
    cov = 1.0 / (n_samples - 1)
    return mean, cov


def get_mc_sigma_points(rng_key, n_features: int, n_samples: int = 100) -> Array:
    """Generate MCMC samples from a normal distribution"""

    sigma_dist = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))

    return sigma_dist.sample((n_samples,), rng_key)
