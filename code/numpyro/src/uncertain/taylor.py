from typing import Tuple

from .moment import MomentTransform
import jax
from chex import Array, dataclass
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp
from .moment import MomentTransform, MomentTransformClass

dist = tfp.distributions


class TaylorO1Transform(MomentTransformClass):
    def __init__(self, gp_pred):
        self.gp_pred = gp_pred

    def _meanf_scaler(self, x):
        return self.gp_pred.predict_mean(x[None, :]).squeeze()

    def _meanf(self, x):
        return self.gp_pred.predict_mean(x)

    def _meandf(self, x):
        return jax.vmap(jax.jacobian(self._meanf_scaler))(x)

    def _varf(self, x, noiseless=True):
        return self.gp_pred.predict_var(x, noiseless=noiseless)

    def _covf(self, x, noiseless=True):
        return self.gp_pred.predict_cov(x, noiseless=noiseless)

    def predict_mean(self, x, x_cov):

        # sigma points
        return self._meanf(x)

    def predict_f(self, x, x_cov, full_covariance=False, noiseless: bool = False):

        # sigma points
        y_mu = self.predict_mean(x, x_cov)
        if full_covariance:

            cov = self.predict_cov(x=x, x_cov=x_cov, noiseless=noiseless)

            return y_mu, cov
        else:
            var = self.predict_var(x=x, x_cov=x_cov, noiseless=noiseless)
            return y_mu, var

    def predict_var(self, x, x_cov, full_covariance=False, noiseless: bool = False):
        # gradient
        dmu_dx = self._meandf(x)

        var_correction = jnp.einsum("ij,jj->i", dmu_dx ** 2, x_cov).reshape(-1, 1)
        # (P,M) = (P,M) - (P, 1)
        var = self._varf(x, noiseless=noiseless)
        var += var_correction
        return var

    def predict_cov(self, x, x_cov, full_covariance=False, noiseless: bool = False):
        # gradient
        dmu_dx = self._meandf(x)

        cov_correction = jnp.einsum("ij,jk,lk->il", dmu_dx, x_cov, dmu_dx)
        cov = self._covf(x).squeeze() + cov_correction.squeeze()

        return cov


class TaylorO2Transform(TaylorO1Transform):
    def __init__(self, gp_pred):
        self.gp_pred = gp_pred

    def _varf_scaler(self, x, noiseless=False):
        return self.gp_pred.predict_var(x[None, :], noiseless=noiseless).squeeze()

    def _meand2f(self, x):
        return jax.vmap(jax.hessian(self._meanf_scaler))(x)

    def _vard2f(self, x, noiseless=False):
        f = jax.partial(self._varf_scaler, noiseless=noiseless)
        return jax.vmap(jax.hessian(f))(x)

    def predict_mean(self, x, x_cov):

        # standard mean
        mu = self._meanf(x)

        # correction (order 2)
        d2mu_dx2 = self._meand2f(x)
        mu_corr = 0.5 * jnp.einsum("ijj,jj->i", d2mu_dx2, x_cov).reshape(-1, 1)

        return mu + mu_corr

    def predict_f(self, x, x_cov, full_covariance=False, noiseless: bool = False):

        # sigma points
        y_mu = self.predict_mean(x, x_cov)
        if full_covariance:

            cov = self.predict_cov(x=x, x_cov=x_cov, noiseless=noiseless)

            return y_mu, cov
        else:
            var = self.predict_var(x=x, x_cov=x_cov, noiseless=noiseless)
            return y_mu, var

    def predict_var(self, x, x_cov, full_covariance=False, noiseless: bool = False):
        # correction (order 1)
        dmu_dx = self._meandf(x)

        var_correction = jnp.einsum("ij,jj->i", dmu_dx ** 2, x_cov).reshape(-1, 1)
        # (P,M) = (P,M) - (P, 1)
        var = self._varf(x, noiseless=noiseless)

        var += var_correction

        # correction (order 2)
        d2var_dx2 = self._vard2f(x)
        var_correction = 0.5 * jnp.einsum("ijj,jj->i", d2var_dx2, x_cov).reshape(-1, 1)

        var += var_correction

        return var

    def predict_cov(self, x, x_cov, full_covariance=False, noiseless: bool = False):
        # correction (order 1)
        dmu_dx = self._meandf(x)

        cov_correction = dmu_dx @ x_cov @ dmu_dx.T
        cov = self._covf(x).squeeze() + cov_correction.squeeze()

        # correction (order 2)
        d2var_dx2 = self._vard2f(x)
        cov_correction = 0.5 * jnp.einsum("ijj,jj,kjj->ik", d2var_dx2, x_cov, d2var_dx2)

        cov += cov_correction.squeeze()

        return cov


def get_mc_weights(n_samples: int = 100) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""
    mean = 1.0 / n_samples
    cov = 1.0 / (n_samples - 1)
    return mean, cov


def get_mc_sigma_points(rng_key, n_features: int, n_samples: int = 100) -> Array:
    """Generate MCMC samples from a normal distribution"""

    sigma_dist = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))

    return sigma_dist.sample((n_samples,), rng_key)
