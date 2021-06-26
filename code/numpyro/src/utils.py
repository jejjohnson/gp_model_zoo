from typing import Tuple, Callable

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from jax.scipy.linalg import cho_factor, cho_solve
from sklearn.cluster import KMeans


def identity_matrix(n_samples: int, constant: float = 1.0) -> Array:
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


def compute_ci_std(var: Array, ci_prct: int = 96) -> Array:

    ci = (100.0 - ci_prct) / 2.0

    return ci * jnp.sqrt(var)


def summarize_posterior(preds, ci=96):
    ci_lower = (100 - ci) / 2
    ci_upper = (100 + ci) / 2
    preds_mean = preds.mean(0)
    preds_lower = jnp.percentile(preds, ci_lower, axis=0)
    preds_upper = jnp.percentile(preds, ci_upper, axis=0)
    return preds_mean, preds_lower, preds_upper


def init_inducing_kmeans(x: Array, n_inducing: int, seed: int = 123, **kwargs) -> Array:
    # conver to numpy array
    x = np.array(x)

    # calculate k-means
    clf = KMeans(n_clusters=n_inducing, random_state=seed, **kwargs).fit(x)

    # convert to jax array
    x_u_init = jnp.array(clf.cluster_centers_)

    return x_u_init


def init_inducing_subsample(x: Array, n_inducing: int, seed: int = 123) -> Array:

    key = jax.random.PRNGKey(seed)
    # random permutation and subsample
    x_u_init = jax.random.permutation(key, x)[:n_inducing]

    return x_u_init


def predict_in_batches(
    Xtest, pred_f, batch_size=5_000, error=True, transform=None, notebook: bool = False
):

    n_batches = max(int(Xtest.shape[0] / batch_size), 1)

    mus, stds = [], []
    if notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    with tqdm(np.array_split(Xtest, n_batches), leave=False) as pbar:
        for X_batch in pbar:

            if transform is not None:
                X_batch = transform(X_batch)
            if error:
                mu_, var_ = pred_f(x=X_batch)
            else:
                mu_, var_ = pred_f(X_batch)
            mus.append(mu_)
            stds.append(np.sqrt(var_))

    # concatenate
    mus = np.concatenate(mus, axis=0)
    stds = np.concatenate(stds, axis=0)

    return mus, stds


def rescale_data(
    x: np.ndarray, transform: Callable, forward: bool = True
) -> np.ndarray:

    if x.ndim < 2:
        x = x[:, None]

    x = transform(x)

    return x
