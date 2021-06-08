import chex
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from .moment import MomentTransform
from chex import Array, dataclass


def init_unscented_transform(
    gp_pred,
    n_features: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    kappa: Optional[float] = None,
):

    wm, wc = get_unscented_weights(n_features, kappa, alpha, beta)

    Wm, Wc = wm, jnp.diag(wc)
    sigma_pts = get_unscented_sigma_points(n_features, kappa, alpha)

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


def get_unscented_sigma_points(
    n_features: int, kappa: Optional[float] = None, alpha: float = 1.0
) -> Tuple[chex.Array, chex.Array]:
    """Generate Unscented samples"""

    # calculate kappa value
    if kappa is None:
        kappa = jnp.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    c = jnp.sqrt(n_features + lam)
    return jnp.hstack(
        (jnp.zeros((n_features, 1)), c * jnp.eye(n_features), -c * jnp.eye(n_features))
    )


def get_unscented_weights(
    n_features: int,
    kappa: Optional[float] = None,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""

    # calculate kappa value
    if kappa is None:
        kappa = jnp.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    wm = 1.0 / (2.0 * (n_features + lam)) * jnp.ones(2 * n_features + 1)
    wc = wm.copy()
    wm = jax.ops.index_update(wm, 0, lam / (n_features + lam))
    wc = jax.ops.index_update(wc, 0, wm[0] + (1 - alpha ** 2 + beta))
    return wm, wc
