from sklearn.base import BaseEstimator, RegressorMixin
from scipy.cluster.vq import kmeans2
import numpy as np
import Gpy


class SGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        n_inducing=10,
        random_state=123,
        max_iters=200,
        optimizer="scg",
        verbose=None,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
        self.rng = np.random.RandomState(random_state)
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]

        # Kernel matrix
        gp_model = GPy.models.SparseGPRegression(X, y, kernel=self.kernel, Z=z)

        # set the fitc inference
        gp_model.inference_method = GPy.inference.latent_function_inference.FITC()

        # Optimize
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model

    def predict(self, X, return_std=False, noiseless=True):

        if noiseless:
            mean, var = self.gp_model.predict_noiseless(X)
        else:
            mean, var = self.gp_model.predict(X)

        if return_std:
            return mean, var
        else:
            return mean

