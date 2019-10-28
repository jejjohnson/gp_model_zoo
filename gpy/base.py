from sklearn.base import BaseEstimator, RegressorMixin
from typing import Optional
import numpy as np


class GPyModel:
    def __init__(self,):
        pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        self.gp_model = None
        return self

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

    def display_model(self):
        return self.gp_model
