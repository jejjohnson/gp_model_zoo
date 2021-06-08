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
from jax.scipy.linalg import cho_solve, cholesky
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam

import wandb
from src.data import small_sample_demo
from src.kernels import rbf_kernel
from src.utils import (
    add_to_diagonal,
    cholesky_factorization,
    cov_to_stddev,
    identity_mat,
)

sns.set_context(context="talk", font_scale=0.7)


wandb.init(project="gp-demo", entity="ipl_uv")
wandb.config.n_train = 100
wandb.config.n_test = 1_000
wandb.config.seed = 123
wandb.config.dataset = "random"
wandb.config.noise = 0.2
wandb.config.model = "exact_gp"
wandb.config.jitter = 1e-5

X, y, Xtest, ytest = small_sample_demo(
    n_train=wandb.config.n_train,
    n_test=wandb.config.n_test,
    noise=wandb.config.noise,
    seed=wandb.config.seed,
)


fig, ax = plt.subplots(ncols=1, figsize=(10, 4))

ax.scatter(X, y, label="Observations", color="red", marker="o")
ax.plot(Xtest, ytest, label="Latent Function", color="black", linewidth=3)
ax.legend()
plt.tight_layout()
wandb.log({f"data": [wandb.Image(plt)]})


def GP(X, y):

    X = numpyro.deterministic("X", X)

    # Set informative priors on kernel hyperparameters.
    η = numpyro.sample("variance", dist.HalfCauchy(scale=5.0))
    ℓ = numpyro.sample("length_scale", dist.Gamma(2.0, 1.0))
    σ = numpyro.sample("obs_noise", dist.HalfCauchy(scale=5.0))

    # Compute kernel
    K = rbf_kernel(X, X, η, ℓ)
    K = add_to_diagonal(K, σ)
    K = add_to_diagonal(K, wandb.config.jitter)
    # cholesky decomposition
    Lff = numpyro.deterministic("Lff", cholesky(K, lower=True))

    # Sample y according to the standard gaussian process formula
    return numpyro.sample(
        "y",
        dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), scale_tril=Lff)
        .expand_by(y.shape[:-1])  # for multioutput scenarios
        .to_event(y.ndim - 1),
        obs=y,
    )


# ===================
# Model
# ===================
# GP model
gp_model = GP

# delta guide - basically deterministic
delta_guide = AutoDelta(GP)


# ===================
# Optimization
# ===================
n_epochs = 2_500
lr = 0.005
optimizer = Adam(step_size=lr)


# ===================
# Training
# ===================
# reproducibility
rng_key = random.PRNGKey(42)

# setup svi
svi = SVI(gp_model, delta_guide, optimizer, loss=Trace_ELBO())

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
# ==================
# Posterior
# ==================
learned_params = delta_guide.median(svi_results.params)


def predict(params, Xtest, noiseless: bool = False):
    # weights

    K = rbf_kernel(X, X, params["variance"], params["length_scale"])
    K = add_to_diagonal(K, params["obs_noise"])
    L, alpha = cholesky_factorization(K, y)

    # projection kernel
    K_xf = rbf_kernel(Xtest, X, params["variance"], params["length_scale"])

    # dot product (mean predictions)
    mu_y = K_xf @ alpha

    # covariance
    v = cho_solve(L, K_xf.T)

    K_xx = rbf_kernel(Xtest, Xtest, params["variance"], params["length_scale"])

    cov_y = K_xx - jnp.dot(K_xf, v)

    if not noiseless:
        cov_y = add_to_diagonal(cov_y, params["obs_noise"])

    return mu_y, cov_y


mu, cov = predict(learned_params, Xtest, noiseless=False)

ci_prnt = 96
ci = 1.96

one_stddev = ci * cov_to_stddev(cov)

fig, ax = plt.subplots(ncols=1, figsize=(10, 4))

ax.scatter(X.ravel(), y.ravel(), color="tab:orange", label="Training Data")
ax.plot(
    Xtest.ravel(), mu.ravel(), color="tab:blue", linewidth=3, label="Predictive Mean"
)
ax.fill_between(
    Xtest.ravel(),
    mu.ravel() - one_stddev.ravel(),
    mu.ravel() + one_stddev.ravel(),
    alpha=0.4,
    color="tab:blue",
    label=f" {ci_prnt}% Confidence Interval",
)
ax.plot(Xtest, mu.ravel() - one_stddev, linestyle="--", color="tab:blue")
ax.plot(Xtest, mu.ravel() + one_stddev, linestyle="--", color="tab:blue")
ax.legend()
plt.tight_layout()
wandb.log({f"predictions": [wandb.Image(plt)]})
