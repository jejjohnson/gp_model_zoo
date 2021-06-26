from chex import Array, dataclass
from typing import Callable, NamedTuple
import abc
from distrax._src.utils.jittable import Jittable


class MomentTransform(NamedTuple):
    predict_mean: Callable
    predict_cov: Callable
    predict_var: Callable
    predict_f: Callable


class MomentTransformClass(Jittable):
    def __init__(self, gp_pred):
        pass

    def predict_f(self, x, x_cov, full_covariance=False):
        raise NotImplementedError()

    def predict_y(self, x, x_cov, full_covariance=False):
        raise NotImplementedError()

    def predict_mean(self, x, x_cov):
        raise NotImplementedError()

    def predict_var(self, x, x_cov):
        raise NotImplementedError()

    def predict_cov(self, x, x_cov):
        raise NotImplementedError()

