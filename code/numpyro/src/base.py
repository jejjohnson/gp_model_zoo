from flax import struct
import jax.numpy as jnp
from chex import dataclass


@dataclass
class Predictive:
    def predict_mean(self, xtest):

        raise NotImplementedError()

    def predict_cov(self, xtest, noiseless=False):

        raise NotImplementedError()

    def _predict(self, xtest, full_covariance: bool = False, noiseless: bool = True):

        raise NotImplementedError()

    def predict_f(self, xtest, full_covariance: bool = False):

        return self._predict(xtest, full_covariance=full_covariance, noiseless=True)

    def predict_y(self, xtest, full_covariance: bool = False):

        return self._predict(xtest, full_covariance=full_covariance, noiseless=False)

    def predict_var(self, xtest, noiseless=False):

        raise jnp.sqrt(self.predict_cov(xtest=xtest, noiseless=noiseless))
