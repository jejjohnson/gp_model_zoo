import GPy
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score

class GPR(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        kernel=None,
        random_state=123,
        max_iters=200,
        optimizer="lbfgs",
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


