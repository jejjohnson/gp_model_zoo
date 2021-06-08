from chex import Array, dataclass


@dataclass
class MomentTransform:
    n_features: int

    def predict_mean(self, X, X_cov):
        raise NotImplementedError()

    def predict_cov(self, X, X_cov):
        raise NotImplementedError()

    def predict_var(self, X, X_cov):
        raise NotImplementedError()

    def predict_f(self, X, X_cov):
        raise NotImplementedError()
