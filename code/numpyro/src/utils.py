from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax.scipy.linalg import cho_factor, cho_solve
from scipy.cluster.vq import kmeans2


def identity_mat(n_samples: int, constant: float = 1.0) -> Array:
    return constant * jnp.eye(n_samples)


def add_to_diagonal(K: Array, constant: float) -> Array:
    return jax.ops.index_add(K, jnp.diag_indices(K.shape[0]), constant)


def cholesky_factorization(K: Array, Y: Array) -> Tuple[Array, bool]:
    """Cholesky Factorization"""

    L = cho_factor(K, lower=True)

    # weights
    # print(L.shape, Y.shape)
    weights = cho_solve(L, Y)

    return L, weights


def cov_to_stddev(cov: Array,) -> Array:

    return jnp.sqrt(jnp.diag(cov))


def compute_ci_bounds(std: Array, ci: int = 96) -> Array:
    ci_lower = (100 - ci) / 2
    ci_upper = (100 + ci) / 2

    return std


def summarize_posterior(preds, ci=96):
    ci_lower = (100 - ci) / 2
    ci_upper = (100 + ci) / 2
    preds_mean = preds.mean(0)
    preds_lower = jnp.percentile(preds, ci_lower, axis=0)
    preds_upper = jnp.percentile(preds, ci_upper, axis=0)
    return preds_mean, preds_lower, preds_upper


def init_inducing_kmeans(x: Array, n_inducing: int) -> Array:
    # conver to numpy array
    x = np.array(x)

    # calculate k-means
    x_u_init = kmeans2(x, n_inducing, minit="points")[0]

    # convert to jax array
    x_u_init = jnp.array(x_u_init)

    return x_u_init


def init_inducing_subsample(key, x: Array, n_inducing: int) -> Array:

    # random permutation and subsample
    x_u_init = jax.random.permutation(key, x)[:n_inducing]

    return x_u_init
