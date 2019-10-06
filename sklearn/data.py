import numpy as np
import pandas
from sklearn.model_selection import train_test_split


def example_1d(func=1, x_error=0.3):
    seed = 123
    rng = np.random.RandomState(seed=seed)

    # sample data parameters
    n_train, n_test, n_trial = 60, 100, 2000
    sigma_y = 0.05
    x_cov = x_error
    x_min, x_max = -10, 10

    # real function
    if func == 1:
        f = lambda x: np.sin(1.0 * np.pi / 1.6 * np.cos(5 + .5 * x))
    elif func == 2:
        f = lambda x: np.sinc(x)

    else:
        f = lambda x: np.sin(2. * x) + np.exp(0.2 * x)

    # Training add x, y = f(x)
    x = np.linspace(x_min, x_max, n_train + n_test)

    x, xs, = train_test_split(x, train_size=n_train, random_state=seed)

    # add noise
    y = f(x)
    x_train = x + x_cov * rng.randn(n_train)
    y_train = f(x) + sigma_y * rng.randn(n_train)

    x_train, y_train = x_train[:, np.newaxis], y_train[:, np.newaxis]

    # -----------------
    # Testing Data
    # -----------------

    ys = f(xs)

    # Add noise
    x_test = xs + x_cov * rng.randn(n_test)
    y_test = ys

    x_test, y_test = x_test[:, np.newaxis], y_test[:, np.newaxis]

    # -------------------
    # Plot Points
    # -------------------
    x_plot = np.linspace(x_min, x_max, n_test)[:, None]
    y_plot = f(x_plot)

    X = {
        'train': x_train,
        'test': x_test,
        'plot': x_plot
    }
    y = {
        'train': y_train,
        'test': y_test,
        'plot': y_plot
    }

    error_params = {
        'x': x_cov,
        'y': sigma_y,
        'f': f
    }

    return X, y, error_params


def main():

    pass

if __name__ == '__main__':
    main()