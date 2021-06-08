from chex import Array, dataclass
from typing import Callable, NamedTuple


class MomentTransform(NamedTuple):
    predict_mean: Callable
    predict_cov: Callable
    predict_var: Callable
    predict_f: Callable
