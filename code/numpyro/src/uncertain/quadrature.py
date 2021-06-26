import chex
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from .moment import MomentTransform
from .unscented import UnscentedTransform
from chex import Array, dataclass
from scipy.special import factorial
from sklearn.utils.extmath import cartesian
from numpy.polynomial.hermite_e import hermegauss, hermeval


class GaussHermiteTransform(UnscentedTransform):
    def __init__(self, gp_pred, n_features: int, degree: int = 3):
        self.gp_pred = gp_pred
        self.wm = get_quadrature_weights(n_features=n_features, degree=degree)
        self.wc = self.wm
        self.Wm, self.Wc = self.wm, jnp.diag(self.wm)
        self.sigma_pts = get_quadrature_sigma_points(
            n_features=n_features, degree=degree
        )


def get_quadrature_sigma_points(n_features: int, degree: int = 3,) -> Array:
    """Generate Unscented samples"""
    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # nD sigma-points by cartesian product
    return cartesian([x] * n_features).T  # column/sigma-point


def get_quadrature_weights(n_features: int, degree: int = 3,) -> Array:
    """Generate normalizers for MCMC samples"""

    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # hermegauss() provides weights that cause posdef errors
    w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
    return jnp.prod(cartesian([w] * n_features), axis=1)
