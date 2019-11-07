import GPy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.metrics import r2_score
from scipy.cluster.vq import kmeans2
from typing import Tuple


class SparseGPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        inference="vfe",
        n_inducing=10,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
        alpha=0.5,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.inference = inference
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:

        # check Array
        X = check_array(X)
        # get dimensions of inputs
        d_dimensions = X.shape[1]

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=True)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Kernel matrix
        self.gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

        # set the fitc inference
        if self.inference.lower() == "vfe":
            self.gp_model.inference_method = (
                GPy.inference.latent_function_inference.VarDTC()
            )

        elif self.inference.lower() == "fitc":
            self.gp_model.inference_method = (
                GPy.inference.latent_function_inference.FITC()
            )

        elif self.inference.lower() == "pep":
            self.gp_model.inference_method = GPy.inference.latent_function_inference.PEP(
                self.alpha
            )
        else:
            raise ValueError(f"Unrecognized inference method: {self.inference}")

        # Make likelihood variance low to start
        self.gp_model.Gaussian_noise.variance = 0.01

        # Optimization
        if self.n_restarts >= 1:
            self.gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True,
                verbose=self.verbose,
                max_iters=self.max_iters,
            )
        else:
            self.gp_model.optimize(
                self.optimizer, messages=self.verbose, max_iters=self.max_iters
            )

        return self

    def display_model(self):
        return self.gp_model

    def predict(
        self, X: np.ndarray, return_std: bool = False, noiseless: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:

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
