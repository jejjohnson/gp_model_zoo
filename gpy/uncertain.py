import GPy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from scipy.cluster.vq import kmeans2



class SparseGPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        X_variance=None,
        inference='vfe',
        n_inducing=10,
        max_iters=200,
        optimizer="scg",
        n_restarts=10,
        verbose=None,
        alpha=.5
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

        # Get inducing points
        z = kmeans2(X, self.n_inducing, minit="points")[0]
        
        # Get Variance
        X_variance = self._check_X_variance(self.X_variance, X)
        

        # Kernel matrix
        gp_model = GPy.models.SparseGPRegression(
            X, y, 
            kernel=self.kernel, 
            Z=z,
            X_variance=X_variance
        )

        # set the fitc inference
        if self.inference.lower() == 'vfe' or X_variance is not None:
            gp_model.inference_method = GPy.inference.latent_function_inference.VarDTC()
            
        elif self.inference.lower() == 'fitc':
            gp_model.inference_method = GPy.inference.latent_function_inference.FITC()
            
        elif self.inference.lower() == 'pep':
            gp_model.inference_method = GPy.inference.latent_function_inference.PEP(self.alpha)
        else:
            raise ValueError(f'Unrecognized inference method: {self.inference}')
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
    
    def _check_X_variance(self, X_variance, X):
        
        if X_variance is None:
            return X_variance
        
        elif isinstance(X_variance, float):
            return X_variance * np.ones(shape=X.shape)
        
        elif isinstance(X_variance, np.ndarray):
            if X_variance.shape == 1:
                return X_variance * np.ones(shape=X.shape)
            elif X_variance.shape == X.shape[1]:
                return np.tile(self.x_variance, (X.shape[0], 1))
            else:
                raise ValueError(f"Shape of 'X_variance' ({X_variance.shape}) "
                                 f"doesn't match X ({X.shape})")
        else:
            raise ValueError(f"Unrecognized type of X_variance.")
    
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
