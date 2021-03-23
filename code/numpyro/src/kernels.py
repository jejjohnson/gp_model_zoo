from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass


# squared euclidean distance
def sqeuclidean_distance(x: Array, y: Array) -> float:
    return jnp.sum((x - y) ** 2)


# distance matrix
def cross_covariance(func: Callable, x: Array, y: Array) -> Array:
    """distance matrix"""
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


# kernel function
def rbf_kernel(X, Y, variance, length_scale):
    # distance formula
    deltaXsq = cross_covariance(
        sqeuclidean_distance, X / length_scale, Y / length_scale
    )

    # rbf function
    K = variance * jnp.exp(-0.5 * deltaXsq)
    return K
