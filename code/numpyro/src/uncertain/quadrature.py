import chex
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from .moment import MomentTransform
from chex import Array, dataclass
from scipy.special import factorial
from sklearn.utils.extmath import cartesian
from numpy.polynomial.hermite_e import hermegauss, hermeval


def init_gausshermite_transform(
    gp_pred,
    n_features: int,
    degree: int = 20,
    kappa: Optional[float] = None,
):

    wm = get_quadrature_weights(n_features=n_features, degree=degree)
    wc = wm
    Wm, Wc = wm, jnp.diag(wm)
    sigma_pts = get_quadrature_sigma_points(n_features=n_features, degree=degree)

    def predict_mean(x, x_cov):

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_sigma = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=1)(
            x_sigma_samples
        )

        # mean of mc samples
        # (N,M,P) @ (M,) -> (N,P)
        y_mu = jnp.einsum("ijk,j->ik", y_mu_sigma, Wm)

        return y_mu

    def predict_f(x, x_cov, full_covariance=False):

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (N,M,P) = (D,M)
        y_mu_sigma = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=1)(
            x_sigma_samples
        )

        # mean of mc samples
        # (N,M,P) @ (M,) -> (N,P)
        y_mu = jnp.einsum("ijk,j->ik", y_mu_sigma, wm)

        # ===================
        # Covariance
        # ===================
        if full_covariance:
            # (N,P,M) - (N,P,1) -> (N,P,M)
            dfydx = y_mu_sigma - y_mu[:, None]

            # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)
            cov = jnp.einsum("ijk,jl,mlk->ikm", dfydx, Wc, dfydx.T)
            return y_mu, cov
        else:

            # (N,P,M) - (N,P,1) -> (N,P,M)
            dfydx = y_mu_sigma - y_mu[:, None]

            # (N,M,P) @ (M,) -> (N,P)
            var = jnp.einsum("ijk,j->ik", dfydx ** 2, wc)

            return y_mu, var

    def predict_cov(x, x_cov):

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (N,P,M) = (D,M)
        y_mu_sigma = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=1)(
            x_sigma_samples
        )

        # mean of mc samples
        # (N,P,M) @ (M,) -> (N,P)
        y_mu = jnp.einsum("ijk,j->ik", y_mu_sigma, Wm)

        # ===================
        # Covariance
        # ===================
        # (N,P,M) - (N,P,1) -> (N,P,M)
        dfydx = y_mu_sigma - y_mu[:, None]

        # (N,M,P) @ (M,M) @ (N,M,P) -> (N,P,D)
        y_cov = jnp.einsum("ijk,jl,mlk->ikm", dfydx, Wc, dfydx.T)

        return y_cov

    def predict_var(x, x_cov):

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (N,P,M) = (D,M)
        y_mu_sigma = jax.vmap(gp_pred.predict_mean, in_axes=2, out_axes=1)(
            x_sigma_samples
        )

        # mean of mc samples
        # (N,P,M) @ (M,) -> (N,P)
        y_mu = jnp.einsum("ijk,j->ik", y_mu_sigma, Wm)

        # ===================
        # Variance
        # ===================
        # (N,P,M) - (N,P,1) -> (N,P,M)
        dfydx = y_mu_sigma - y_mu[:, None]

        # (N,M,P) @ (M,) -> (N,P)
        var = jnp.einsum("ijk,j->ik", dfydx ** 2, wc)

        return var

    return MomentTransform(
        predict_mean=predict_mean,
        predict_cov=predict_cov,
        predict_f=predict_f,
        predict_var=predict_var,
    )


def get_quadrature_sigma_points(
    n_features: int,
    degree: int = 3,
) -> Array:
    """Generate Unscented samples"""
    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # nD sigma-points by cartesian product
    return cartesian([x] * n_features).T  # column/sigma-point


def get_quadrature_weights(
    n_features: int,
    degree: int = 3,
) -> Array:
    """Generate normalizers for MCMC samples"""

    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # hermegauss() provides weights that cause posdef errors
    w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
    return jnp.prod(cartesian([w] * n_features), axis=1)
