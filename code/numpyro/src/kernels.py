from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass
from flax import struct


@struct.dataclass
class Kernel:
    def cross_covariance(self, X, Y):
        return NotImplementedError

    def gram(self, X):
        return self.cross_covariance(X, X)

    def diag(self, X):
        return NotImplementedError


@struct.dataclass
class RBF(Kernel):
    variance: Array
    length_scale: Array

    def cross_covariance(self, X, Y):
        # distance formula
        deltaXsq = cross_covariance(
            sqeuclidean_distance, X / self.length_scale, Y / self.length_scale
        )

        # rbf function
        K = self.variance * jnp.exp(-0.5 * deltaXsq)
        return K

    def diag(self, X):
        return self.variance * jnp.ones(X.shape[0])


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
