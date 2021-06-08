import jax.numpy as jnp


def zero_mean(X):
    return jnp.zeros(X.shape[0])
