import numpy as np
from numpy.polynomial.hermite import hermgauss


def integrate_hermgauss_nd(func, mean, sigma, order):
    """
    n-d Gauss-Hermite quadrature

    Parameters
    ----------
    func : a callable function that will handle the predictions

    mean : d_dimensions
        A d-dimensional sample

    sigma_x : d_dimensions
        A d-dimensional matrix of covariances

    order : the accuracy of the gaussian approximation

    :param order: the order of the integration rule
    :return: :math:`E[f(X)] (X \sim \mathcal{N}(\mu,\sigma^2)) = \int_{-\infty}^{\infty}f(x)p(x),\mathrm{d}x` with p being the normal density
    """
    from itertools import product
    # if np.ndim(mean) == 1:
    # 	mean = mean[:, np.newaxis]


    dim = len(mean) 			# Dimensions
    mean = np.array(mean)

    # Create diagonal matrix for sigma
    sigma = np.sqrt(sigma)
    x, w = hermgauss(order)
    xs = product(x, repeat=dim)
    ws = np.array(list(product(w,repeat=dim)))
    y = []
    sqrt2 = np.sqrt(2)
    for i,x in enumerate(xs):
        input = x*sigma*sqrt2+mean
        y.append(func(input[np.newaxis, :])*ws[i].prod())
    y = np.array(y)
    return y.sum() / np.sqrt(np.pi)**dim # * 1/(sigma*np.sqrt(2*np.pi)) * sigma * np.sqrt(2)