import GPy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from scipy.cluster.vq import kmeans2

class SVGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        n_inducing=10,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
        batch_size=100,
    ):
        self.kernel = kernel
        self.n_inducing = n_inducing
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

        # GP Model w. MiniBatch support
        gp_model = GPy.models.sparse_gp_minibatch.SparseGPMiniBatch(
            X,
            y,
            kernel=self.kernel,
            Z=z,
            likelihood=GPy.likelihoods.Gaussian(),
            batchsize=self.batch_size,
            stochastic=False,
            missing_data=False,
            inference_method=GPy.inference.latent_function_inference.VarDTC()
        )
            
        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01
        
        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True, 
                verbose=self.verbose,
                max_iters=self.max_iters
            )
        else:
            gp_model.optimize(
                self.optimizer, 
                messages=self.verbose, 
                max_iters=self.max_iters
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
            return mean, np.sqrt(var)
        else:
            return mean

class VGP(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
    ):
        self.kernel = kernel
        self.max_iters = max_iters
        self.optimizer = optimizer
        self.n_restarts = n_restarts
        self.verbose = verbose

    def fit(self, X, y):

        n_samples, d_dimensions = X.shape

        # default Kernel Function
        if self.kernel is None:
            self.kernel = GPy.kern.RBF(input_dim=d_dimensions, ARD=False)

        # VGP Model
        gp_model = GPy.models.GPVariationalGaussianApproximation(
            X, 
            y, 
            kernel=self.kernel, 
            likelihood=GPy.likelihoods.Gaussian()
        )
            
        # Make likelihood variance low to start
        gp_model.Gaussian_noise.variance = 0.01
        
        # Optimization
        if self.n_restarts >= 1:
            gp_model.optimize_restarts(
                num_restarts=self.n_restarts,
                robust=True, 
                verbose=self.verbose,
                max_iters=self.max_iters
            )
        else:
            gp_model.optimize(
                self.optimizer, 
                messages=self.verbose, 
                max_iters=self.max_iters
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
            return mean, np.sqrt(var)
        else:
            return mean