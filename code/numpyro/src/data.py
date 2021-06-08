import jax
import jax.numpy as jnp


def regression_simple(
    n_train: int = 50, n_test: int = 1_000, noise: float = 0.2, seed: int = 123, buffer: float=0.1
):
    key = jax.random.PRNGKey(seed)
    x = (
        jax.random.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n_train,))
        .sort()
        .reshape(-1, 1)
    )
    f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x)
    signal = f(x)
    y = signal + jax.random.normal(key, shape=signal.shape) * noise
    xtest = jnp.linspace(-3.0 - buffer, 3.0 + buffer, n_test).reshape(-1, 1)
    ytest = f(xtest)
    return x, y, xtest, ytest


def regression_complex(
    n_train: int = 1_000, n_test: int = 1_000, noise: float = 0.2, seed: int = 123, buffer: float=0.1
):
    key = jax.random.PRNGKey(seed)

    x = jnp.linspace(-1.0, 1.0, n_train).reshape(-1, 1)  # * 2.0 - 1.0
    f = (
        lambda x: jnp.sin(x * 3 * 3.14)
        + 0.3 * jnp.cos(x * 9 * 3.14)
        + 0.5 * jnp.sin(x * 7 * 3.14)
    )

    signal = f(x)
    y = signal + noise * jax.random.normal(key, shape=signal.shape)

    xtest = jnp.linspace(-1.0 - buffer, 1.0 + buffer, n_test).reshape(-1, 1)
    ytest = f(xtest)
    return x, y, xtest, ytest
