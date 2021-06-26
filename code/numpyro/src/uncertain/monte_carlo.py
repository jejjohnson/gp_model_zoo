from typing import Tuple

from .moment import MomentTransform, MomentTransformClass
import jax
from chex import Array, dataclass
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions
import abc
from distrax._src.utils.jittable import Jittable


class MCTransform(MomentTransformClass):
    def __init__(self, gp_pred, n_samples: int, cov_type: bool = "diag"):
        self.gp_pred = gp_pred
        self.n_samples = n_samples
        if cov_type == "diag":
            self.z = dist.MultivariateNormalDiag
        elif cov_type == "full":
            self.z = dist.MultivariateNormalFullCovariance
        else:
            raise ValueError(f"Unrecognized covariance type: {cov_type}")
        self.wm, self.wc = get_mc_weights(n_samples)

    def predict_f(self, key, x, x_cov, full_covariance=False):

        # create distribution
        # print(x.min(), x.max(), x_cov.min(), x_cov.max())
        x_dist = self.z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((self.n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(self.gp_pred.predict_mean, in_axes=0, out_axes=1)(
            x_mc_samples
        )

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        if full_covariance:
            # ===================
            # Covariance
            # ===================
            # (N,P,M) - (N,P,1) -> (N,P,M)
            dfydx = y_mu_mc - y_mu[..., None]

            # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)

            Wc = jnp.eye(self.n_samples) * self.wc
            cov = jnp.einsum("ijk,jl,mlk->ikm", dfydx, Wc, dfydx.T)

            # cov = self.wc * jnp.einsum("ijk,lmn->ilk", dfydx, dfydx)

            return y_mu, cov
        else:
            # (N,P) = (N,M,P)
            var = jnp.var(y_mu_mc, axis=1)

            return y_mu, var

    def predict_mean(self, key, x, x_cov):

        # create distribution
        x_dist = self.z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((self.n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(self.gp_pred.predict_mean, in_axes=0, out_axes=1)(
            x_mc_samples
        )

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        return y_mu

    def predict_cov(self, key, x, x_cov):

        # create distribution
        x_dist = self.z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((self.n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(self.gp_pred.predict_mean, in_axes=0, out_axes=1)(
            x_mc_samples
        )

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)
        # ===================
        # Covariance
        # ===================
        # (N,P,M) - (N,P,1) -> (N,P,M)
        dfydx = y_mu_mc - y_mu[..., None]

        # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)
        cov = self.wc * jnp.einsum("ijk,lmn->ikl", dfydx, dfydx.T)

        return y_mu, cov

    def predict_var(self, key, x, x_cov):

        # create distribution
        x_dist = self.z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((self.n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(self.gp_pred.predict_mean, in_axes=0, out_axes=1)(
            x_mc_samples
        )

        # variance of mc samples
        # (N,P,) = (N,M,P)
        y_var = jnp.var(y_mu_mc, axis=1)

        return y_var


def init_mc_transform(gp_pred, n_samples: int, cov_type: bool = "diag"):

    if cov_type == "diag":
        z = dist.MultivariateNormalDiag
    elif cov_type == "full":
        z = dist.MultivariateNormalFullCovariance
    else:
        raise ValueError(f"Unrecognized covariance type: {cov_type}")
    wm, wc = get_mc_weights(n_samples)

    def predict_mean(key, x, x_cov):

        # create distribution
        x_dist = z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=0, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        return y_mu

    def predict_f(key, x, x_cov, full_covariance=False):

        # create distribution
        # print(x.min(), x.max(), x_cov.min(), x_cov.max())
        x_dist = z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=0, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        if full_covariance:
            # ===================
            # Covariance
            # ===================
            # (N,P,M) - (N,P,1) -> (N,P,M)
            dfydx = y_mu_mc - y_mu[..., None]

            # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)
            cov = wc * jnp.einsum("ijk,lmn->ikl", dfydx, dfydx.T)

            return y_mu, cov
        else:
            # (N,P) = (N,M,P)
            var = jnp.var(y_mu_mc, axis=1)
            return y_mu, var

    def predict_cov(key, x, x_cov):

        # create distribution
        x_dist = z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=0, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_mu = jnp.mean(y_mu_mc, axis=1)
        # ===================
        # Covariance
        # ===================
        # (N,P,M) - (N,P,1) -> (N,P,M)
        dfydx = y_mu_mc - y_mu[..., None]

        # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)
        cov = wc * jnp.einsum("ijk,lmn->ikl", dfydx, dfydx.T)

        return y_mu, cov

    def predict_var(key, x, x_cov):

        # create distribution
        x_dist = z(x, x_cov)

        # sample
        x_mc_samples = x_dist.sample((n_samples,), key)

        # function predictions over mc samples
        # (N,M,P) = f(N,D,M)
        y_mu_mc = jax.vmap(gp_pred.predict_mean, in_axes=0, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (N,P,) = (N,M,P)
        y_var = jnp.var(y_mu_mc, axis=1)

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
