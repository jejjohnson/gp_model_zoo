import jax
import jax.numpy as np
import time
from numpyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO


def run_nuts_inference(model, rng_key, X, Y, n_warmup, n_samples, n_chains):
    kernel = NUTS(model)
    mcmc = MCMC(kernel, n_warmup, n_samples, num_chains=n_chains, progress_bar=True)
    mcmc.run(rng_key, X, Y)
    mcmc.print_summary()
    return mcmc.get_samples()


def run_svi_inference(model, guide, rng_key, X, Y, optimizer, n_epochs=1_000):

    # initialize svi
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # initialize state
    init_state = svi.init(rng_key, X, Y.squeeze())

    # Run optimizer for 1000 iteratons.
    state, losses = jax.lax.scan(
        lambda state, i: svi.update(state, X, Y.squeeze()), init_state, n_epochs
    )

    # Extract surrogate posterior.
    params = svi.get_params(state)

    return params
