from typing import Tuple

from .moment import MomentTransform
import jax
from chex import Array, dataclass
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions


def init_taylor_o1_transform(gp_pred):
    meanf = lambda x: gp_pred.predict_mean(x)
    varf = lambda x: gp_pred.predict_var(x, noiseless=True)
    vary = lambda x: gp_pred.predict_var(x, noiseless=False)
    covf = lambda x: gp_pred.predict_cov(x)
    _f = lambda x: gp_pred.predict_mean(x[None, :]).squeeze()
    meandf = jax.vmap(jax.grad(_f))

    def predict_mean(x, x_cov):

        return meanf(x)

    def predict_f(x, x_cov, full_covariance=False, noiseless: bool = False):

        # sigma points
        y_mu = meanf(x)

        # gradient
        dmu_dx = meandf(x)

        # correction
        cov_correction = dmu_dx @ x_cov @ dmu_dx.T

        if full_covariance:

            cov = covf(x) + cov_correction

            return y_mu, cov
        else:
            # (P,M) = (P,M) - (P, 1)
            if noiseless:
                var = varf(x)
            else:
                var = vary(x)

            var += jnp.diagonal(cov_correction).reshape(-1, 1)
            return y_mu, var

    def predict_cov(x, x_cov):

        # gradient
        dmu_dx = meandf(x)

        # correction
        cov_correction = dmu_dx @ x_cov @ dmu_dx.T

        cov = covf(x) + cov_correction

        return cov

    def predict_var(x, x_cov):

        raise NotImplementedError()

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
