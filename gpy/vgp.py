from sklearn.base import BaseEstimator, RegressorMixin
from scipy.cluster.vq import kmeans2
import numpy as np
import GPy


class VGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        random_state=123,
        max_iters=200,
        optimizer="scg",
        verbose=None,
    ):
        self.kernel = kernel
        self.rng = np.random.RandomState(random_state)
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.verbose = verbose

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # likelihood function
        likelihood = GPy.likelihoods.Gaussian()

        # VGP Model
        gp_model = GPy.models.GPVariationalGaussianApproximation(
            X, y, kernel=self.kernel, likelihood=likelihood
        )

        # Optimize
        # gp_model.inducing_inputs.fix()
        gp_model.optimize(
            self.optimizer, messages=self.verbose, max_iters=self.max_iters
        )

        self.gp_model = gp_model

        return self

    def display_model(self):
        return self.gp_model
