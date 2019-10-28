import GPy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from scipy.cluster.vq import kmeans2
from typing import Optional, Union


class UnscentedGPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        X_variance=None,
        random_state=123,
        max_iters=200,
        optimizer="lbfgs",
        n_restarts=10,
        verbose=None,
    ):
        self.kernel = kernel
        self.X_variance = X_variance
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose

    def fit(self, X, y):

        # get shapes
        n_samples, d_dimensions = X.shape

        # check X_variance
        self.X_variance = self._check_X_variance(self.X_variance, d_dimensions)

        # default Kernel Function
        if self.kernel is None:
            kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)
        else:
            kernel = self.kernel

        # Kernel matrix
        gp_model = GPy.models.GPRegression(X, y, kernel)

        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def _check_X_variance(
        self, X_variance: Union[None, float, np.ndarray], n_dims: int
    ) -> None:

        if X_variance is None:
            return X_variance

        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=(n_dims, n_dims))

        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.identity(n_dims)
            elif X_variance.shape == n_dims:
                return np.diag(X_variance)
            else:
                raise ValueError(
                    f"Shape of 'X_variance' ({X_variance.shape}) "
                    f"doesn't match X ({n_dims})"
                )
        else:
            raise ValueError(f"Unrecognized type of X_variance.")

    def display_model(self):
        return self.gp_model

    def predict(
        self, X, return_std=False, full_cov=False, noiseless=True, unscented=True
    ):

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        if return_std:
            # we want the variance correction
            if unscented == True and self.X_variance is not None:
                # get the variance correction
                var_add = self._variance_correction(X)

                # get diagonal elements only
                if full_cov == False:
                    var_add = np.diag(var_add)[:, None]

                # add correction to original variance
                var += var_add

                # return mean and standard deviation
                return mean, np.sqrt(var)

            else:
                return mean, np.sqrt(var)
        else:
            return mean

    def _variance_correction(self, X: np.ndarray) -> np.ndarray:
        # calculate the gradient
        x_der, _ = self.gp_model.predictive_gradients(X)

        # calculate correction
        var_add = x_der[..., 0] @ self.X_variance @ x_der[..., 0].T
        return var_add


class UnscentedSGPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        inference="vfe",
        X_variance=None,
        n_inducing=10,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
        alpha=0.5,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.X_variance = X_variance
        self.inference = inference
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.alpha = alpha

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # check X_variance
        self.X_variance = self._check_X_variance(self.X_variance, d_dimensions)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Kernel matrix
        gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

        # set the fitc inference
        if self.inference.lower() == "vfe":
            gp_model.inference_method = GPy.inference.latent_function_inference.VarDTC()

        elif self.inference.lower() == "fitc":
            gp_model.inference_method = GPy.inference.latent_function_inference.FITC()

        elif self.inference.lower() == "pep":
            gp_model.inference_method = GPy.inference.latent_function_inference.PEP(
                self.alpha
            )
        else:
            raise ValueError(f"Unrecognized inference method: {self.inference}")
        # Optimize
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01

        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model

    def _check_X_variance(
        self, X_variance: Union[None, float, np.ndarray], n_dims: int
    ) -> None:

        if X_variance is None:
            return X_variance

        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=(n_dims, n_dims))

        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.identity(n_dims)
            elif X_variance.shape == n_dims:
                return np.diag(X_variance)
            else:
                raise ValueError(
                    f"Shape of 'X_variance' ({X_variance.shape}) "
                    f"doesn't match X ({n_dims})"
                )
        else:
            raise ValueError(f"Unrecognized type of X_variance.")

    def predict(
        self, X, return_std=False, full_cov=False, noiseless=True, unscented=True
    ):

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        if return_std:
            # we want the variance correction
            if unscented == True and self.X_variance is not None:
                # get the variance correction
                var_add = self._variance_correction(X)

                # get diagonal elements only
                if full_cov == False:
                    var_add = np.diag(var_add)[:, None]
                # add correction to original variance
                var += var_add

                # return mean and standard deviation
                return mean, np.sqrt(var)

            else:
                return mean, np.sqrt(var)
        else:
            return mean

    def _variance_correction(self, X: np.ndarray) -> np.ndarray:
        # calculate the gradient
        x_der, _ = self.gp_model.predictive_gradients(X)

        # calculate correction
        var_add = x_der[..., 0] @ self.X_variance @ x_der[..., 0].T
        return var_add


class UncertainSGPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        X_variance=None,
        inference="vfe",
        n_inducing=10,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
        batch_size=None,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.X_variance = X_variance
        self.inference = inference
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.batch_size = batch_size

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Get Variance
        X_variance = self._check_X_variance(self.X_variance, X)

        # Inference function
        if self.inference.lower() == "vfe" or X_variance is not None:
            inference_method = GPy.inference.latent_function_inference.VarDTC()

        elif self.inference.lower() == "fitc":
            inference_method = GPy.inference.latent_function_inference.FITC()

        else:
            raise ValueError(f"Unrecognized inference method: {self.inference}")

        # Kernel matrix

        if self.batch_size is None:
            gp_model = GPy.models.SparseGPRegression(
                X, y, kernel=self.kernel, Z=z, X_variance=X_variance
            )
        else:
            gp_model = GPy.models.bayesian_gplvm_minibatch.BayesianGPLVMMiniBatch(
                Y=y,
                X=X,
                input_dim=X.shape,
                kernel=self.kernel,
                Z=z,
                X_variance=X_variance,
                inference_method=inference_method,
                batchsize=self.batch_size,
                likelihood=GPy.likelihoods.Gaussian(),
                stochastic=False,
                missing_data=False,
            )

        # set the fitc inference

        # Optimize
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01

        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        self.gp_model = gp_model

        return self

    def _check_X_variance(self, X_variance, X):

        if X_variance is None:
            return X_variance

        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=X.shape)

        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.ones(shape=X.shape)
            elif X_variance.shape == X.shape[1]:
                return np.tile(self.X_variance, (X.shape[0], 1))
            else:
                raise ValueError(
                    f"Shape of 'X_variance' ({X_variance.shape}) "
                    f"doesn't match X ({X.shape})"
                )
        else:
            raise ValueError(f"Unrecognized type of X_variance.")

    def display_model(self):
        return self.gp_model

    def predict(self, X, return_std=False, noiseless=True):

        if noiseless == True:
            include_likelihood = False
        elif noiseless == False:
            include_likelihood = True
        else:
            raise ValueError(f"Unrecognized argument for noiseless: {noiseless}")

        mean, var = self.gp_model.predict(X, include_likelihood=include_likelihood)

        if return_std:
            return mean, np.sqrt(var)
        else:
            return mean

