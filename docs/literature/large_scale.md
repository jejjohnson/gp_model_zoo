---
title: Large Scale
description: GPs and seriously scaling to large data
authors:
    - J. Emmanuel Johnson
path: docs/literature/
source: large_scale.md
---
# GPs at Large Scale

> Any literature related to scaling GPs to astronomical size. 1 million points and higher. Also things with respectable orders of magnitude.


??? abstract "Fast Direct Methods for Gaussian Processes by Ambikasaran et. al. (2014)"
    > Scales GPs with dimensions greater than 3. 

    -> [Paper](https://arxiv.org/abs/1403.6015)

    -> [Code](https://george.readthedocs.io/en/latest/)

??? tip "Constant-Time Predictive Distributions for GPs - Pleiss et. al. (2018)"
    > Using MVM techniquees, they are able to make constant time predictive mean and variance estimates; addressing the computational bottleneck of predictive distributions for GPs.

    -> [Paper](https://arxiv.org/abs/1803.06058)
    
    -> [Code]()

??? abstract "When Gaussian Process Meets Big Data: A Review of Scalable GPs - (2019)"

    > A great review paper on GPs and how to scale them. Goes over most of the SOTA.

    -> [Paper](https://arxiv.org/abs/1807.01065)

??? tip "Exact GP on a Million Data Points - Wang et. al. (2019)"
    > The authors manage to train a GP with multiple GPUs on a million or so data points. They use the matrix-vector-multiplication strategy. Quite amazing actually...

    * [Paper](https://arxiv.org/abs/1903.08114)
    * [Code](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Simple_GP_Regression/Simple_MultiGPU_GP_Regression.ipynb)

??? info "Randomly Projected Additive Gaussian Processes for Regression - Delbridge et. al. (12-2019)"
    -> [Paper](https://arxiv.org/abs/1912.12834)

    -> [Code](https://github.com/idelbrid/Randomly-Projected-Additive-GPs) (**PyTorch**)

??? info "Efficiently Sampling Functions from Gaussian Process Posteriors - Wilson et. al. (16-2020)"
    > Uses a path-wise sampling scheme to efficiently sample for GP posteriors. Motivates the use for GP priors for monte carlo estimation schemes.
    
    -> [Tweet 1](https://twitter.com/avt_im/status/1281266970025959425) | [Tweet 2](https://twitter.com/avt_im/status/1326125319078686721?s=20) 

    -> [Blog](https://sml-group.cc/blog/2020-gp-sampling/)

    -> [Video](https://icml.cc/virtual/2020/poster/6461)

    -> [Paper](https://arxiv.org/abs/2002.09309)
    

    - Code - [![Julia](https://raw.githubusercontent.com/JuliaLang/julia/master/doc/src/assets/logo.svg){:height="25px" width="40px"}](https://github.com/aterenin/SparseGaussianProcesses.jl) | [![GPFlow](https://github.com/GPflow/GPflow/raw/develop/doc/source/_static/gpflow_logo.svg){:height="25px" width="60px"}](https://github.com/j-wilson/GPflowSampling)



??? tip "Fast Matrix Square Roots with Applications to Gaussian Processes and Bayesian Optimization by Pleiss et. al. (2020)"
    -> [Paper](https://arxiv.org/abs/2006.11267)

    -> [Code](https://docs.gpytorch.ai/en/v1.2.0/examples/02_Scalable_Exact_GPs/Exact_GP_Posterior_Sampling_with_CIQ.html)

??? tip "Sparse Cholesky factorization by Kullback-Leibler minimization - Schafer et, al. (2020)"
    -> [Paper](https://arxiv.org/abs/2004.14455)