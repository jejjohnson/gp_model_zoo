from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from chex import Array
from jax import random
from jax.scipy.linalg import cholesky, solve_triangular
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam

import wandb
from src.data import large_sample_demo
from src.kernels import rbf_kernel
from src.utils import (
    add_to_diagonal,
    cholesky_factorization,
    cov_to_stddev,
    identity_mat,
    init_inducing_kmeans,
)

sns.set_context(context="talk", font_scale=0.7)


wandb.init(project="gp-demo", entity="ipl_uv")
wandb.config.n_train = 1_000
wandb.config.n_inducing = 20
wandb.config.n_test = 1_000
wandb.config.seed = 123
wandb.config.dataset = "random"
wandb.config.noise = 0.2
wandb.config.model = "sparse_gp"
wandb.config.inducing_init = "kmeans"
wandb.config.jitter = 1e-5


X, y, Xtest, ytest = large_sample_demo(
    n_train=wandb.config.n_train,
    n_test=wandb.config.n_test,
    noise=wandb.config.noise,
    seed=wandb.config.seed,
)


X_u_init = init_inducing_kmeans(X, wandb.config.n_inducing)


jitter = wandb.config.jitter


def SparseGP(X, y):

    n_samples = X.shape[0]
    X = numpyro.deterministic("X", X)
    # Set priors on kernel hyperparameters.
    η = numpyro.sample("variance", dist.HalfCauchy(scale=5.0))
    ℓ = numpyro.sample("length_scale", dist.Gamma(2.0, 1.0))
    σ = numpyro.sample("obs_noise", dist.HalfCauchy(scale=5.0))

    x_u = numpyro.param("x_u", init_value=X_u_init)

    # η = numpyro.param("kernel_var", init_value=1.0, constraints=dist.constraints.positive)
    # ℓ = numpyro.param("kernel_length", init_value=0.1,  constraints=dist.constraints.positive)
    # σ = numpyro.param("sigma", init_value=0.1, onstraints=dist.constraints.positive)

    # ================================
    # Mean Function
    # ================================
    f_loc = np.zeros(n_samples)

    # ================================
    # Qff Term
    # ================================
    # W   = (inv(Luu) @ Kuf).T
    # Qff = Kfu @ inv(Kuu) @ Kuf
    # Qff = W @ W.T
    # ================================
    Kuu = rbf_kernel(x_u, x_u, η, ℓ)
    Kuf = rbf_kernel(x_u, X, η, ℓ)
    # Kuu += jnp.eye(Ninducing) * jitter
    # add jitter
    Kuu = add_to_diagonal(Kuu, jitter)

    # cholesky factorization
    Luu = cholesky(Kuu, lower=True)
    Luu = numpyro.deterministic("Luu", Luu)

    # W matrix
    W = solve_triangular(Luu, Kuf, lower=True)
    W = numpyro.deterministic("W", W).T

    # ================================
    # Likelihood Noise Term
    # ================================
    # D = noise
    # ================================
    D = numpyro.deterministic("G", jnp.ones(n_samples) * σ)

    # ================================
    # trace term
    # ================================
    # t = tr(Kff - Qff) / noise
    # t /= - 2.0
    # ================================
    Kffdiag = jnp.diag(rbf_kernel(X, X, η, ℓ))
    Qffdiag = jnp.power(W, 2).sum(axis=1)
    trace_term = (Kffdiag - Qffdiag).sum() / σ
    trace_term = jnp.clip(trace_term, a_min=0.0)  # numerical errors

    # add trace term to the log probability loss
    numpyro.factor("trace_term", -trace_term / 2.0)

    # Sample y according SGP
    return numpyro.sample(
        "y",
        dist.LowRankMultivariateNormal(loc=f_loc, cov_factor=W, cov_diag=D)
        .expand_by(y.shape[:-1])
        .to_event(y.ndim - 1),
        obs=y,
    )


# ===================
# Model
# ===================
# GP model
sgp_model = SparseGP

# delta guide - basically deterministic
delta_guide = AutoDelta(SparseGP)


# ===================
# Optimization
# ===================
n_epochs = 1_000
lr = 0.01
optimizer = Adam(step_size=lr)


# ===================
# Training
# ===================
# reproducibility
rng_key = random.PRNGKey(42)

# setup svi
svi = SVI(sgp_model, delta_guide, optimizer, loss=Trace_ELBO())

# run svi
svi_results = svi.run(rng_key, n_epochs, X, y.T)

# ===================
# Plot Loss
# ===================
fig, ax = plt.subplots(ncols=1, figsize=(6, 4))
ax.plot(svi_results.losses)
ax.set(title="Loss", xlabel="Iterations", ylabel="Negative Log-Likelihood")
plt.tight_layout()
wandb.log({f"loss": [wandb.Image(plt)]})

wandb.log({f"nll_loss": np.array(svi_results.losses[-1])})
learned_params = delta_guide.median(svi_results.params)
learned_params["x_u"] = svi_results.params["x_u"]

# =================
# Predictions
# =================
from copy import deepcopy


def get_cond_params(
    learned_params: dict, x: Array, y: Array, jitter: float = 1e-5
) -> dict:

    params = deepcopy(learned_params)
    n_samples = x.shape[0]

    # calculate the cholesky factorization
    Kuu = rbf_kernel(
        params["x_u"], params["x_u"], params["variance"], params["length_scale"]
    )
    Kuu = add_to_diagonal(Kuu, jitter)
    Luu = cholesky(Kuu, lower=True)

    Kuf = rbf_kernel(params["x_u"], x, params["variance"], params["length_scale"])

    W = solve_triangular(Luu, Kuf, lower=True)
    D = np.ones(n_samples) * params["obs_noise"]

    W_Dinv = W / D
    K = W_Dinv @ W.T
    K = add_to_diagonal(K, 1.0)

    L = cholesky(K, lower=True)

    # mean function
    y_residual = y  # mean function
    y_2D = y_residual.reshape(-1, n_samples).T
    W_Dinv_y = W_Dinv @ y_2D

    return {"Luu": Luu, "W_Dinv_y": W_Dinv_y, "L": L}


def _pred_factorize(params, xtest):

    Kux = rbf_kernel(params["x_u"], xtest, params["variance"], params["length_scale"])
    Ws = solve_triangular(params["Luu"], Kux, lower=True)
    # pack
    pack = jnp.concatenate([params["W_Dinv_y"], Ws], axis=1)
    Linv_pack = solve_triangular(params["L"], pack, lower=True)
    # unpack
    Linv_W_Dinv_y = Linv_pack[:, : params["W_Dinv_y"].shape[1]]
    Linv_Ws = Linv_pack[:, params["W_Dinv_y"].shape[1] :]

    return Ws, Linv_W_Dinv_y, Linv_Ws


n_test = Xtest.shape[0]

cond_params = get_cond_params(learned_params, X, y.T)

cond_params = {**learned_params, **cond_params}


def sparse_predict_mean(cond_params: dict, xtest: Array) -> Array:

    _, Linv_W_Dinv_y, Linv_Ws = _pred_factorize(cond_params, xtest)
    loc_shape = y.T.shape[:-1] + (n_test,)
    loc = (Linv_W_Dinv_y.T @ Linv_Ws).reshape(loc_shape)

    return loc.T


def sparse_predict_covariance(
    params: dict, xtest: Array, noiseless: bool = False
) -> Array:

    n_test_samples = xtest.shape[0]

    Ws, _, Linv_Ws = _pred_factorize(cond_params, xtest)

    Kxx = rbf_kernel(xtest, xtest, params["variance"], params["length_scale"])

    if not noiseless:
        Kxx = add_to_diagonal(Kxx, params["obs_noise"])

    Qss = Ws.T @ Ws

    cov = Kxx - Qss + Linv_Ws.T @ Linv_Ws

    cov_shape = y.T.shape[:-1] + (n_test_samples, n_test_samples)
    cov = np.reshape(cov, cov_shape)

    return cov


mu = sparse_predict_mean(cond_params, Xtest)
cov = sparse_predict_covariance(cond_params, Xtest, noiseless=False)


ci_prnt = 96
ci = 1.96
one_stddev = ci * jnp.sqrt(np.diag(cov.squeeze()))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(X, y.squeeze(), "o", color="tab:orange", label="Training Data")
ax.plot(Xtest, mu.ravel(), color="tab:blue", linewidth=3, label="Predictive Mean")
ax.fill_between(
    Xtest.ravel(),
    mu.ravel() - one_stddev,
    mu.ravel() + one_stddev,
    alpha=0.4,
    color="tab:blue",
    label=f" {ci_prnt}% Confidence Interval",
)
plt.scatter(
    cond_params["x_u"],
    np.zeros_like(cond_params["x_u"]),
    label="Inducing Points",
    color="black",
    marker="x",
    zorder=3,
)
ax.plot(Xtest, mu.ravel() - one_stddev, linestyle="--", color="tab:blue")
ax.plot(Xtest, mu.ravel() + one_stddev, linestyle="--", color="tab:blue")
plt.tight_layout()
plt.legend()
wandb.log({f"predictions": [wandb.Image(plt)]})
