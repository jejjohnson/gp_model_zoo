import jax
import jax.numpy as jnp


def regression_near_square(
    n_train: int = 50,
    n_test: int = 1_000,
    x_noise: float = 0.3,
    y_noise: float = 0.2,
    seed: int = 123,
    buffer: float = 0.1,
):
    key = jax.random.PRNGKey(seed)

    # function
    f = lambda x: jnp.sin(1.0 * jnp.pi / 1.6 * jnp.cos(5 + 0.5 * x))

    # input training data (clean)
    xtrain = jnp.linspace(-10, 10, n_train).reshape(-1, 1)
    ytrain = f(xtrain) + jax.random.normal(key, shape=xtrain.shape) * y_noise
    xtrain_noise = xtrain + x_noise * jax.random.normal(key, shape=xtrain.shape)

    # output testing data (noisy)
    xtest = jnp.linspace(-10.0 - buffer, 10.0 + buffer, n_test)[:, None]
    ytest = f(xtest)
    xtest_noise = xtest + x_noise * jax.random.normal(key, shape=xtest.shape)

    idx_sorted = jnp.argsort(xtest_noise, axis=0)
    xtest_noise = xtest_noise[idx_sorted[:, 0]]
    ytest_noise = ytest[idx_sorted[:, 0]]

    return xtrain, xtrain_noise, ytrain, xtest, xtest_noise, ytest, ytest_noise


def regression_simple(
    n_train: int = 50,
    n_test: int = 1_000,
    noise: float = 0.2,
    seed: int = 123,
    buffer: float = 0.1,
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
    n_train: int = 1_000,
    n_test: int = 1_000,
    noise: float = 0.2,
    seed: int = 123,
    buffer: float = 0.1,
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
