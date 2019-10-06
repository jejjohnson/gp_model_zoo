import warnings
from operator import itemgetter
import numba
from numba import prange
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated


class EGP(GaussianProcessRegressor):
    def __init__(self, x_variance=None, n_restarts=5, kernel=None):
        if kernel is None:
            kernel = C() * RBF() + WhiteKernel()

        super().__init__(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=123,
        )

        self.x_variance = self._build_variance(x_variance)
        self._vK_inv = None

    def _build_variance(self, x_variance=None):
        # Fix the Variance of X
        if x_variance is None:
            x_variance = 0.0
        if isinstance(x_variance, float):
            x_variance = np.array(x_variance).reshape(1, -1)
        if np.ndim(x_variance) < 2:
            x_variance = np.diag(x_variance)
        return x_variance

    def _build_variance_weights(self):

        # ======================================
        # Step I: Take Derivative
        # ======================================

        # Calculate the Derivative for RBF Kernel
        self.derivative = rbf_derivative(
            self.X_train_,
            self.X_train_,
            self.kernel_(self.X_train_, self.X_train_),
            self.alpha_,
            self.kernel_.get_params()["k1__k2__length_scale"],
        )

        # Calculate the derivative term
        self.derivative_term = np.dot(
            self.derivative, np.dot(self.x_variance, self.derivative.T)
        )

        # ======================================
        # Step II: Find Weights
        # ======================================

        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        K += self.derivative_term

        try:
            self._vL = cholesky(K, lower=True)

        except np.linalg.LinAlgError as exc:
            exc.args(
                f"The kernel {self.kernel_}, is not returing a "
                "positive definite matrix. Try gradually "
                "increasing the 'alpha' parameter of your GPR."
            ) + exc.args
            raise

        self.variance_alpha_ = cho_solve((self._vL, True), self.y_train_)

        L_inv = solve_triangular(self._vL.T, np.eye(self._vL.shape[0]))
        self._vK_inv = L_inv.dot(L_inv.T)
        return self

    def predict(self, X, return_std=False, error_variance=False):
        """Predict using the Gaussian process regression model
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, also its
        standard deviation (return_std=True) or covariance (return_cov=True).
        Note that at most one of the two can be requested.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Query points where the GP is evaluated
        return_std : bool, default: False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default: False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean
        Returns
        -------
        y_mean : array, shape = (n_samples, [n_output_dims])
            Mean of predictive distribution a query points
        y_std : array, shape = (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when return_std is True.
        y_cov : array, shape = (n_samples, n_samples), optional
            Covariance of joint predictive distribution a query points.
            Only returned when return_cov is True.
        """
        X = check_array(X)

        K_trans = self.kernel_(X, self.X_train_)

        y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
        y_mean = self._y_train_mean + y_mean  # undo normal.

        if return_std:
            if error_variance:
                return y_mean, np.sqrt(self.evariance(X, K_trans))
            else:
                return y_mean, np.sqrt(self.variance(X, K_trans))
        else:
            return y_mean

    def variance(self, X, K_trans=None):
        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)

        if self._K_inv is None:
            # compute inverse K_inv of K based on its Cholesky
            # decomposition L and its inverse L_inv
            L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
            self._K_inv = L_inv.dot(L_inv.T)

        # Compute variance of predictive distribution
        y_var = self.kernel_.diag(X)
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. " "Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0
        return y_var

    def evariance(self, X, K_trans=None, error_variance=False):

        if self._vK_inv is None:

            self._build_variance_weights()

        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)

        length_scale = self.kernel_.get_params()["k1__k2__length_scale"]
        derivative = rbf_derivative(
            self.X_train_,
            X,
            weights=self.variance_alpha_,
            K=K_trans,
            length_scale=length_scale,
        )
        derivative_term = np.einsum(
            "ij,ij->i", np.dot(derivative, self.x_variance), derivative
        )

        # Compute variance of predictive distribution
        y_var = self.kernel_.diag(X) + derivative_term
        # print(K_trans.shape, self._vK_inv)
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, self._vK_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. " "Setting those variances to 0."
            )
            y_var[y_var_negative] = 0.0
        return y_var

    def predict_noiseless(self, X, return_std=True, error_variance=False):

        if return_std:
            self.bias = np.sqrt(self.kernel_.k2.noise_level)
            mean, std = self.predict(X, return_std=True, error_variance=error_variance)
            std -= self.bias
            return mean, std
        else:
            mean = self.predict(X, return_std=False, error_variance=error_variance)
            return mean





def rbf_derivative(x_train, x_function, K, weights, length_scale):
    """The derivative of the RBF kernel. It returns the derivative
    as a 2D matrix.

    Parameter
    ---------
    xtrain : array, (n_train_samples x d_dimensions)

    xtest : array, (n_test_samples x d_dimensions)

    K : array (n_test_samples, n_train_samples)

    weights : array, (ntrain_samples)

    length_scale : float

    Return
    ------

    Derivative : array, (n_test, d_dimensions)

    Information
    -----------
    Name : J. Emmanuel Johnson
    Date
    """
    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    for itest in range(n_test):

        term1 = (np.expand_dims(x_function[itest, :], axis=0) - x_train).T
        term2 = K[itest, :] * weights.squeeze()

        derivative[itest, :] = np.dot(term1, term2)

    derivative *= -1 / length_scale ** 2

    return derivative

@numba.njit(fastmath=True, nogil=True)
def uncertain_variance_numba(xtrain, xtest, K, Kinv, weights, mu, 
                             signal_variance, length_scale, x_cov):
    
    # calculate the determinant constant
    det_term = 2 * x_cov * np.power(length_scale, -2) + 1
    det_term = 1 / np.sqrt(np.linalg.det(np.diag(det_term)))
    
    # calculate the exponential scale
    exp_scale = np.power(length_scale, 2) + 0.5 * np.power(length_scale, 4) * np.power(x_cov, -1)
    exp_scale = np.power(exp_scale, -1)
    
    # Calculate the constants
    y_var = signal_variance - mu**2
    
    n_test = xtest.shape[0]
    
    for itest in range(n_test):
        qi = calculate_q_numba(xtrain, xtest[itest, :], K[:, itest], det_term, exp_scale)
        y_var[itest] -= np.trace(np.dot(Kinv, qi))
        y_var[itest] += np.dot(weights.T, np.dot(qi, weights))[0][0]
    
    
    
    return np.sqrt(y_var)

@numba.jit(nopython=True, nogil=True)
def calculate_q_numba(x_train, x_test, K, det_term, exp_scale):
    """Calculates the Q matrix used to compute the variance of the
    inputs with a noise covariance matrix. This uses numba to 
    speed up the calculations.
    
    Parameters
    ----------
    x_train : array, (n_samples x d_dimensions)
        The data used to train the weights.
    
    x_test : array, (d_dimensions)
        A vector of test points.
        
    K : array, (n_samples)
        The portion of the kernel matrix of the training points at 
        test point i, e.g. K = full_kernel_mat[:, i_test]
        
    det_term : float
        The determinant term that's in from of the exponent
        term.
        
    exp_scale : array, (d_dimensions)
        The length_scale that's used within the exponential term.
        
    Returns
    -------
    Q : array, (n_samples x n_samples)
        The Q matrix used to calculate the variance of the samples.
        
    Information
    -----------
    Author : J. Emmanuel Johnson
    Email  : jemanjohnson34@gmail.com
    Date   : 13 - 06 - 2018
    
    References
    ----------
    McHutchen et al. - Gaussian Process Training with Input Noise
    http://mlg.eng.cam.ac.uk/pub/pdf/MchRas11.pdf
    """
    n_train, d_dimensions = x_train.shape
    
    Q = np.zeros(shape=(n_train, n_train), dtype=np.float64)
    
    # Loop through the row terms
    for iterrow in range(n_train):
        
        # Calculate the row terms
        x_train_row = 0.5 * x_train[iterrow, :]  - x_test
        
        K_row = K[iterrow] * det_term
        
        # Loop through column terms
        for itercol in range(n_train):
            
            # Z Term
            z_term = x_train_row + 0.5 * x_train[itercol, :]
            
            # EXPONENTIAL TERM
            exp_term = np.exp( np.sum( z_term**2 * exp_scale) )
            
            # CONSTANT TERM
            constant_term = K_row * K[itercol] 
            
            # Q Matrix (Corrective Gaussian Kernel)
            Q[iterrow, itercol] = constant_term * exp_term
            
    return Q

@numba.njit(fastmath=True, nogil=True)
def ard_derivative_numba(x_train, x_function, K, weights, length_scale):
    #     # check the sizes of x_train and x_test
    #     err_msg = "xtrain and xtest d dimensions are not equivalent."
    #     np.testing.assert_equal(x_function.shape[1], x_train.shape[1], err_msg=err_msg)

    #     # check the n_samples for x_train and weights are equal
    #     err_msg = "Number of training samples for xtrain and weights are not equal."
    #     np.testing.assert_equal(x_train.shape[0], weights.shape[0], err_msg=err_msg)

    n_test, n_dims = x_function.shape

    derivative = np.zeros(shape=x_function.shape)

    length_scale = np.diag(- np.power(length_scale, -2))

    for itest in prange(n_test):
        derivative[itest, :] = np.dot(np.dot(length_scale, (x_function[itest, :] - x_train).T),
                                      (K[itest, :].reshape(-1, 1) * weights))

    return derivative


def ard_weighted_covariance(X, Y=None, x_cov=None, length_scale=None,
                            signal_variance=None):

    # grab samples and dimensions
    n_samples, n_dimensions = X.shape

    # get the default sigma values
    if length_scale is None:
        length_scale = np.ones(shape=n_dimensions)

    # check covariance values
    if x_cov is None:
        x_cov = np.array([0.0])

    # Add dimensions to lengthscale and x_cov
    if np.ndim(length_scale) == 0:
        length_scale = np.array([length_scale])

    if np.ndim(x_cov) == 0:
        x_cov = np.array([x_cov])

    # get default scale values
    if signal_variance is None:
        signal_variance = 1.0

    exp_scale = np.sqrt(x_cov + length_scale ** 2)

    scale_term = np.diag(x_cov * (length_scale ** 2) ** (-1)) + np.eye(N=n_dimensions)
    scale_term = np.linalg.det(scale_term)
    scale_term = signal_variance * np.power(scale_term, -1 / 2)


    # Calculate the distances
    D = np.expand_dims(X / exp_scale, 1) - np.expand_dims(Y / exp_scale, 0)

    # Calculate the kernel matrix
    K = scale_term * np.exp(-0.5 * np.sum(D ** 2, axis=2))

    return K

