import warnings
from operator import itemgetter

import sys
sys.path.insert(0, '/Users/eman/Documents/code_projects/kernellib')

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated
from sklearn.datasets import make_regression
from kernellib.derivatives import ard_derivative
from gp_error.data import example_1d
import matplotlib.pyplot as plt
from gp_error.uncertainty.GaussianProcess import GaussianProcess as GPError
from gp_error.uncertainty.Covariance import GaussianCovariance
from gp_error.uncertainty.UncertaintyPropagation import UncertaintyPropagationNumericalHG
from gp_error.batch import gperror_parallel
from gp_error.utils import integrate_hermgauss_nd


class GPErrorTaylorApprox(object):
    def __init__(self, gp_model, x_error=None):
        self.gp_model = gp_model
        if x_error is None:
            x_error = 0.0
        if isinstance(x_error, float):
            x_error = np.array([x_error])
        self.x_error = np.diag(x_error)

    def fit(self):

        #######################################
        # Modifications for the Variance Term
        #######################################
        self.signal_variance = self.gp_model.kernel_.get_params()['k1__k1__constant_value']
        self.length_scale = [1 / self.gp_model.kernel_.get_params()['k1__k2__length_scale']**2]
        self.noise_likelihood = self.gp_model.kernel_.get_params()['k2__noise_level']
        self.X_train_ = self.gp_model.X_train_
        self.y_train_ = self.gp_model.y_train_
        self.alpha_ = self.gp_model.alpha_
        # Calculate the Inverse Covariance matrix

        L_inv = solve_triangular(self.gp_model.L_.T, np.eye(self.gp_model.L_.shape[0]))
        self.K_inv = L_inv.dot(L_inv.T)
        self.Kinvt = np.dot(self.K_inv, self.y_train_)

        return self

    def predict(self, X, return_std=False):

        K_traintest = self.gp_model.kernel_(self.X_train_, X)

        n_samples, d_dimensions = np.shape(self.gp_model.X_train_)
        mu = np.dot(K_traintest.T, self.alpha_)

        if not return_std:
            return mu
        else:
            S2 = self.variance(X, K_traintest.T)

            for iteration, ix in enumerate(X):
                xtest = np.atleast_2d(ix)

                deriv = np.zeros(shape=(1, d_dimensions))
                S2deriv = np.zeros(shape=(d_dimensions, d_dimensions))

                for idim in range(d_dimensions):
                    tmp = (self.X_train_[:, idim] - xtest[:, idim]).flatten()
                    c = (K_traintest[:, iteration].flatten() * tmp)[:, np.newaxis]
                    deriv[:, idim] = self.length_scale[idim] * np.dot(c.T, self.Kinvt)
                    ainvKc = K_traintest[:, iteration][:, np.newaxis] * np.dot(self.K_inv, c)

                    for jdim in range(d_dimensions):
                        exp_t = - self.length_scale[idim] * self.length_scale[jdim]
                        tmp = (self.X_train_[:, idim] - xtest[:, idim]).flatten()[:, np.newaxis]
                        S2deriv[idim, jdim] = exp_t * np.sum(ainvKc * tmp)

                    S2deriv[idim, idim] = S2deriv[idim, idim] + self.length_scale[idim] * self.signal_variance

                S2[iteration] += deriv.dot(self.x_error).dot(deriv.T) + 0.5 * np.sum(np.sum(self.x_error * S2deriv))

            return mu, np.sqrt(S2)

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.gp_model.kernel_(X, self.gp_model.X_train_)

        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(self.gp_model.L_.T, np.eye(self.gp_model.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = self.gp_model.kernel_.diag(X)
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0

        return y_var



class GPErrorVar(object):
    def __init__(self, gp_model, x_error=None):
        self.gp_model = gp_model
        if x_error is None:
            x_error = 0.0
        if isinstance(x_error, float):
            x_error = np.array([x_error])
        self.x_error = np.diag(x_error)

    def fit(self):

        #######################################
        # Modifications for the Variance Term
        #######################################
        self.signal_variance = self.gp_model.kernel_.get_params()['k1__k1__constant_value']
        self.length_scale = self.gp_model.kernel_.get_params()['k1__k2__length_scale']
        derivative = ard_derivative(self.gp_model.X_train_, self.gp_model.X_train_,
                                    weights=self.gp_model.alpha_,
                                    length_scale=self.length_scale,
                                    scale=self.signal_variance)
        self.derivative_train_ = derivative
        # derivative_term = np.diag(np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T))))

        derivative_term = np.einsum("ij,ij->i", np.dot(derivative, self.x_error), derivative)
        K = self.gp_model.kernel_(self.gp_model.X_train_)
        K[np.diag_indices_from(K)] += self.gp_model.alpha + derivative_term
        L = np.linalg.cholesky(K)
        L_inv = solve_triangular(self.gp_model.L_.T, np.eye(self.gp_model.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)

        self.derivative_train_ = derivative
        self.K_inv = K_inv

        return self

    def predict(self, X, return_std=None):

        # Predict based on GP posterior
        K_trans = self.gp_model.kernel_(X, self.gp_model.X_train_)
        y_mean = K_trans.dot(self.gp_model.alpha_)  # Line 4 (y_mean = f_star)
        y_mean = self.gp_model._y_train_mean + y_mean  # undo normal.


        if return_std:

            y_var = self.variance(X, K_trans)

            return y_mean, np.sqrt(y_var)
        else:
            return y_mean

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.gp_model.kernel_(X, self.gp_model.X_train_)
        ########################
        # Derivative
        ########################
        derivative = ard_derivative(self.gp_model.X_train_, X, weights=self.gp_model.alpha_,
                                    length_scale=self.length_scale,
                                    scale=self.signal_variance)
        derivative_term = np.einsum("ij,ij->i", np.dot(derivative, self.x_error), derivative)
        # print(derivative_term.shape)
        # derivative_term = np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T)))
        # print(derivative_term.shape)
        # print(X.shape)
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(self.gp_model.L_.T, np.eye(self.gp_model.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = self.gp_model.kernel_.diag(X) + derivative_term
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
        return y_var

class GPErrorVariance(BaseEstimator, RegressorMixin):
    """Gaussian process regression (GPR).
    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.
    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:
       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.
    Read more in the :ref:`User Guide <gaussian_process>`.
    .. versionadded:: 0.18
    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)
    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)
    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters
    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``
    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    """
    def __init__(self, x_covariance=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        self.x_covariance = x_covariance
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    @property
    @deprecated("Attribute rng was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def rng(self):
        return self._rng

    @property
    @deprecated("Attribute y_train_mean was deprecated in version 0.19 and "
                "will be removed in 0.21.")
    def y_train_mean(self):
        return self._y_train_mean

    def fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        Returns
        -------
        self : returns an instance of self.
        """

        if self.x_covariance is None:
            warnings.warn("No covariance matrix found. Using standard GP...")
            self.x_covariance = 0.0 * np.eye(X.shape[1])
        else:
            if isinstance(self.x_covariance, float):
                self.x_covariance = np.array([self.x_covariance])
            if np.ndim(self.x_covariance) == 1:
                self.x_covariance = self.x_covariance[:, np.newaxis]
            else:
                self.x_covariance = np.diag(self.x_covariance)



        # Define the kernel
        self.kernel_ = C() * RBF() + WhiteKernel()


        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # Calculate the Derivative

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        #######################################
        # Modifications for the Variance Term
        #######################################
        self.signal_variance = self.kernel_.get_params()['k1__k1__constant_value']
        self.length_scale = self.kernel_.get_params()['k1__k2__length_scale']
        derivative = ard_derivative(self.X_train_, self.X_train_, weights=self.alpha_, length_scale=self.length_scale,
                                    scale=self.signal_variance)
        self.derivative_train_ = derivative
        # derivative_term = np.diag(np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T))))

        derivative_term = np.diag(np.einsum("ij,ij->i", np.dot(derivative, self.x_covariance), derivative))

        L = np.linalg.cholesky((K + derivative_term))
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)

        self.derivative_train_ = derivative
        self.K_inv = K_inv

        return self

    def predict(self, X, return_std=False, return_cov=False):
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
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = self.kernel_ = C() * RBF() + WhiteKernel()

                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.

            if return_cov:
                y_cov = self.cov(X)
                return y_mean, y_cov
            elif return_std:

                y_var = self.variance(X, K_trans)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)
        ########################
        # Derivative
        ########################
        derivative = ard_derivative(self.X_train_, X, weights=self.alpha_,
                                    length_scale=self.length_scale,
                                    scale=self.signal_variance)
        derivative_term = np.einsum("ij,ij->i", np.dot(derivative, self.x_covariance), derivative)
        # print(derivative_term.shape)
        # derivative_term = np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T)))
        # print(derivative_term.shape)
        # print(X.shape)
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = self.kernel_.diag(X) + derivative_term
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
        return y_var



    def cov(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)

        v = cho_solve((self.L_, True), K_trans.T)  # Line 5
        y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

        return y_cov


    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.
        Parameters
        ----------
        X : array-like, shape = (n_samples_X, n_features)
            Query points where the GP samples are evaluated
        n_samples : int, default: 1
            The number of samples drawn from the Gaussian process
        random_state : int, RandomState instance or None, optional (default=0)
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the
            random number generator; If None, the random number
            generator is the RandomState instance used by `np.random`.
        Returns
        -------
        y_samples : array, shape = (n_samples_X, [n_output_dims], n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """
        rng = check_random_state(random_state)

        y_mean, y_cov = self.predict(X, return_cov=True)
        if y_mean.ndim == 1:
            y_samples = rng.multivariate_normal(y_mean, y_cov, n_samples).T
        else:
            y_samples = \
                [rng.multivariate_normal(y_mean[:, i], y_cov,
                                         n_samples).T[:, np.newaxis]
                 for i in range(y_mean.shape[1])]
            y_samples = np.hstack(y_samples)
        return y_samples

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        kernel = self.kernel_.clone_with_theta(theta)

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        K[np.diag_indices_from(K)] += self.alpha
        try:
            L = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros_like(theta)) \
                if eval_gradient else -np.inf

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        alpha = cho_solve((L, True), y_train)  # Line 3

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension
            tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min





class GPRegressor(GaussianProcessRegressor, BaseEstimator, RegressorMixin):
    """Gaussian process regression (GPR).
    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.
    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:
       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.
    Read more in the :ref:`User Guide <gaussian_process>`.
    .. versionadded:: 0.18
    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.
    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::
            'fmin_l_bfgs_b'
    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.
    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.
    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)
    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Target values in training data (also required for prediction)
    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters
    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``
    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    """
    def __init__(self, x_covariance=None, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        super(GaussianProcessRegressor, self).__init__()
        # self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state
        self.x_covariance = x_covariance



    def fit(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        Returns
        -------
        self : returns an instance of self.
        """

        if self.x_covariance is None:
            warnings.warn("No covariance matrix found. Using standard GP...")
            self.x_covariance = 0.0 * np.eye(X.shape[1])
        else:
            if isinstance(self.x_covariance, float):
                self.x_covariance = np.array([self.x_covariance])
            if np.ndim(self.x_covariance) == 1:
                self.x_covariance = self.x_covariance[:, np.newaxis]
            else:
                self.x_covariance = np.diag(self.x_covariance)



        # Define the kernel
        self.kernel_ = C() * RBF() + WhiteKernel()


        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        # Calculate the Derivative

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta)

            theta = self.kernel_.theta
            init_weights = np.zeros(self.X_train_.shape[0])
            new_theta = np.concatenate([theta, init_weights])
            # print(theta.shape, new_theta.shape)
            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        #######################################
        # Modifications for the Variance Term
        #######################################
        self.signal_variance = self.kernel_.get_params()['k1__k1__constant_value']
        self.length_scale = self.kernel_.get_params()['k1__k2__length_scale']
        derivative = ard_derivative(self.X_train_, self.X_train_, weights=self.alpha_, length_scale=self.length_scale,
                                    scale=self.signal_variance)
        self.derivative_train_ = derivative
        # derivative_term = np.diag(np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T))))

        derivative_term = np.diag(np.einsum("ij,ij->i", np.dot(derivative, self.x_covariance), derivative))

        L = np.linalg.cholesky((K + derivative_term))
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)

        self.derivative_train_ = derivative
        self.K_inv = K_inv

        return self

    def predict(self, X, return_std=False, return_cov=False):
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
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        X = check_array(X)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = self.kernel_ = C() * RBF() + WhiteKernel()

                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

        else:  # Predict based on GP posterior
            K_trans = self.kernel_(X, self.X_train_)
            y_mean = K_trans.dot(self.alpha_)  # Line 4 (y_mean = f_star)
            y_mean = self._y_train_mean + y_mean  # undo normal.

            if return_cov:
                y_cov = self.cov(X)
                return y_mean, y_cov
            elif return_std:

                y_var = self.variance(X, K_trans)

                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def variance(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)
        ########################
        # Derivative
        ########################
        derivative = ard_derivative(self.X_train_, X, weights=self.alpha_,
                                    length_scale=self.length_scale,
                                    scale=self.signal_variance)
        derivative_term = np.einsum("ij,ij->i", np.dot(derivative, self.x_covariance), derivative)
        # print(derivative_term.shape)
        # derivative_term = np.diag(np.dot(derivative, np.dot(self.x_covariance, derivative.T)))
        # print(derivative_term.shape)
        # print(X.shape)
        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        K_inv = L_inv.dot(L_inv.T)
        # Compute variance of predictive distribution
        y_var = self.kernel_.diag(X) + derivative_term
        y_var -= np.einsum("ij,ij->i", np.dot(K_trans, K_inv), K_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
        return y_var

    def cov(self, X, K_trans=None):

        if K_trans is None:
            K_trans = self.kernel_(X, self.X_train_)

        v = cho_solve((self.L_, True), K_trans.T)  # Line 5
        y_cov = self.kernel_(X) - K_trans.dot(v)  # Line 6

        return y_cov



class GPErrorNumerical(object):
    def __init__(self, gp_model, x_error=None, n_jobs=2, batch_size=20,
                 verbose=0, order=4):
        self.gp_model = gp_model
        if x_error is None:
            x_error = 0.0
        if isinstance(x_error, float):
            x_error = np.array([x_error])

        self.x_error = x_error
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.order = order

    def fit(self, X, y):

        pass

    def predict(self, X, return_std=False):

        means = self.propagate_mean(X, self.order)

        if not return_std:
            return means
        else:
            return means, self.propagate_variance(X, self.order)


    def propagate_mean(self, u, order=4):
        mean_function = lambda x: self.gp_model.predict(x)
        means = list()
        for iu in u:
            means.append(integrate_hermgauss_nd(mean_function, iu, self.x_error, order))

        return np.array(means)

    def propagate_variance(self, u, order=4):
        mu = self.propagate_mean(u)

        mean_function = lambda x: self.gp_model.predict(x)**2
        var_function = lambda x: self.gp_model.predict(x, return_std=True)[1]

        term1 = list()
        term2 = list()

        for iu in u:
            term1.append(integrate_hermgauss_nd(var_function, iu, self.x_error, order))
            term2.append(integrate_hermgauss_nd(mean_function, iu, self.x_error, order))

        term1 = np.array(term1)
        term2 = np.array(term2)
        # print(term2, mu**2)
        return term1 + term2 - mu**2



def test_variance_err():
    X, y, error_params = example_1d()


    # My GP
    gp_model = GPErrorVariance()
    gp_model.fit(X['train'], y['train'])
    mean, std = gp_model.predict(X['test'], return_std=True)
    # Their GP
    kernel = C() * RBF() + WhiteKernel()
    sk_gp_model = GaussianProcessRegressor(kernel=kernel)
    sk_gp_model.fit(X['train'], y['train'])
    sk_mean, sk_std = sk_gp_model.predict(X['test'], return_std=True)

    # Check initial solutions
    np.testing.assert_array_equal(mean, sk_mean)
    np.testing.assert_array_equal(std, sk_std)

    pass

def main():

    pass

if __name__ == '__main__':
    main()