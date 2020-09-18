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

??? abstract "When Gaussian Process Meets Big Data: A Review of Scalable GPs - (2019)"

    > A great review paper on GPs and how to scale them. Goes over most of the SOTA.

    -> [Paper](https://arxiv.org/abs/1807.01065)

??? tip "Exact GP on a Million Data Points - Wang et. al. (2019)"
    > The authors manage to train a GP with multiple GPUs on a million or so data points. They use the matrix-vector-multiplication strategy. Quite amazing actually...

    * [Paper](https://arxiv.org/abs/1903.08114)
    * [Code](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Simple_GP_Regression/Simple_MultiGPU_GP_Regression.ipynb)

??? tip "Constant-Time Predictive Distributions for GPs - Pleiss et. al. (2018)"
    > Using MVM techniquees, they are able to make constant time predictive mean and variance estimates; addressing the computational bottleneck of predictive distributions for GPs.

    * [Paper](https://arxiv.org/abs/1803.06058)
    * [Code]()

